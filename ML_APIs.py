import os
from dotenv import load_dotenv
import cv2
import torch
import numpy as np
import mysql.connector
from mtcnn import MTCNN
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import faiss
from tqdm import tqdm
import random
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import base64
import uuid
from mysql.connector import Error
from pydantic import BaseModel
from io import BytesIO
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import io
import asyncio
import json
import httpx

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "TRUE"  # Allows multiple OpenMP runtimes (temporary fix)
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH")
HEAD_POSE_MODEL_PATH = os.getenv("HEAD_POSE_MODEL_PATH")

os.environ.update(
    {
        "DB_HOST": DB_HOST,
        "DB_PORT": DB_PORT,
        "DB_DATABASE": DB_DATABASE,
        "DB_USERNAME": DB_USERNAME,
        "DB_PASSWORD": DB_PASSWORD,
    }
)

# Load YOLOv8 face detection model
face_detector = YOLO(FACE_MODEL_PATH)

# Load head pose estimation model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head_pose_model_path = HEAD_POSE_MODEL_PATH


class FaceDB:
    def __init__(self):
        self.conn = None
        self._verify_env_vars()
        self._init_db()

    def _verify_env_vars(self):
        required_vars = ["DB_HOST", "DB_DATABASE", "DB_USERNAME", "DB_PASSWORD"]
        missing = [var for var in required_vars if var not in os.environ]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")

    def _get_connection(self):
        try:
            conn = mysql.connector.connect(
                host=os.environ["DB_HOST"],
                database=os.environ["DB_DATABASE"],
                user=os.environ["DB_USERNAME"],
                password=os.environ["DB_PASSWORD"],
                port=int(os.environ.get("DB_PORT", "3306")),
            )
            print("✅ Connected to MySQL")
            return conn
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def _init_db(self):
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT UNSIGNED NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_id ON face_embeddings(user_id)
            """
            )
            self.conn.commit()
            print("✅ Database initialized")
        except Error as e:
            print(f"Error initializing database: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if self.conn and self.conn.is_connected():
                self.conn.close()

    def store_embeddings(self, user_id, embeddings):
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            for emb in embeddings:
                emb_norm = emb / np.linalg.norm(emb)
                cursor.execute(
                    "INSERT INTO face_embeddings (user_id, embedding) VALUES (%s, %s)",
                    (user_id, emb_norm.tobytes()),
                )
            self.conn.commit()
            print(f"✅ Stored embeddings for user {user_id}")
        except Error as e:
            print(f"Error storing embeddings: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if self.conn and self.conn.is_connected():
                self.conn.close()

    def load_embeddings(self, user_id=None):
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor(dictionary=True)
            query = """
            SELECT u.id as user_id, u.name, fe.embedding 
            FROM face_embeddings fe
            JOIN users u ON fe.user_id = u.id
        """

            params = []
            if user_id is not None:
                query += "WHERE fe.user_id = %s"
                params.append(user_id)

            cursor.execute(query, params)

            embeddings = []
            labels = []
            user_ids = []
            for row in cursor:
                embeddings.append(np.frombuffer(row["embedding"], dtype=np.float32))
                labels.append(row["name"])
                user_ids.append(row["user_id"])
            return np.array(embeddings), np.array(labels), np.array(user_ids)
        except Error as e:
            print(f"Error loading embeddings: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if self.conn and self.conn.is_connected():
                self.conn.close()

    def get_student_names(self):
        try:
            self.conn = self._get_connection()
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM users")
            names = [row[0] for row in cursor.fetchall()]
            return names
        except Error as e:
            print(f"Error getting student names: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if self.conn and self.conn.is_connected():
                self.conn.close()


class FaceRecognition:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
        self.detector = MTCNN()
        self.facenet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.align_params = {
            "desired_left_eye": (0.35, 0.35),
            "desired_right_eye": (0.65, 0.35),
            "desired_nose": (0.50, 0.50),
        }

    def _align_face(self, img, keypoints):
        src_points = np.array(
            [keypoints["left_eye"], keypoints["right_eye"], keypoints["nose"]],
            dtype=np.float32,
        )
        dst_points = np.array(
            [
                [
                    self.target_size[0] * self.align_params["desired_left_eye"][0],
                    self.target_size[1] * self.align_params["desired_left_eye"][1],
                ],
                [
                    self.target_size[0] * self.align_params["desired_right_eye"][0],
                    self.target_size[1] * self.align_params["desired_right_eye"][1],
                ],
                [
                    self.target_size[0] * self.align_params["desired_nose"][0],
                    self.target_size[1] * self.align_params["desired_nose"][1],
                ],
            ],
            dtype=np.float32,
        )
        M = cv2.getAffineTransform(src_points, dst_points)
        return cv2.warpAffine(img, M, self.target_size)

    def process_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_rgb[:, :, 0] = clahe.apply(img_rgb[:, :, 0])
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_LAB2RGB)
        detections = self.detector.detect_faces(img_rgb)
        if not detections:
            return None
        return self._align_face(img_rgb, detections[0]["keypoints"])

    def generate_embeddings(self, face_img):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        face_tensor = torch.from_numpy(face_img.transpose(2, 0, 1)).float().to(device)
        face_tensor = (face_tensor / 255.0 - 0.5) / 0.5
        with torch.no_grad():
            return self.facenet(face_tensor.unsqueeze(0)).cpu().numpy().flatten()


class BatchProcessor:
    def __init__(self):
        self.fr = FaceRecognition()
        self.db = FaceDB()

    def _augment(self, face):
        augments = []
        for _ in range(5):
            hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = hsv[..., 1] * random.uniform(0.8, 1.2)
            hsv[..., 2] = hsv[..., 2] * random.uniform(0.8, 1.2)
            aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            if random.random() > 0.5:
                aug = np.fliplr(aug)
            if random.random() > 0.3:
                noise = np.random.normal(0, 0.05, aug.shape)
                aug = np.clip(aug + noise, 0, 255)
            augments.append(aug)
        return augments

    def process_student(self, directory, user_id):
        all_embeddings = []
        image_files = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # Verify that the user_id exists in the users table
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"User with ID {user_id} does not exist")
        except Error as e:
            print(f"Error verifying user_id: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

        # Process images and generate embeddings
        for img_file in tqdm(image_files[:500], desc=f"Processing user_id {user_id}"):
            img_path = os.path.join(directory, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                aligned = self.fr.process_image(img)
                if aligned is None:
                    continue
                for aug_face in self._augment(aligned):
                    embedding = self.fr.generate_embeddings(aug_face)
                    all_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

        # Store embeddings if any were generated
        if all_embeddings:
            self.db.store_embeddings(user_id, all_embeddings)
            print(f"✅ User {user_id} stored with {len(all_embeddings)} embeddings")
        else:
            print(f"❌ Failed to process user {user_id}")


class RealTimeRecognizer:
    def __init__(self):
        self.yolo = YOLO(FACE_MODEL_PATH)
        self.fr = FaceRecognition()
        self.db = FaceDB()
        self._refresh_embeddings()

    def _refresh_embeddings(self, user_id=None):
        """Update embeddings from the database"""
        self.embeddings, self.labels, self.user_ids = self.db.load_embeddings(user_id)
        if len(self.embeddings) == 0:
            self.index = None
            print(
                f"No embeddings found for user_id: {user_id if user_id else 'all users'}"
            )
            return
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print(f"✅ Loaded {len(self.embeddings)} embeddings")

    def process_image(self, img, user_id=None):
        """Process image for API usage"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.yolo(img_rgb)
        matches = []

        if results[0].boxes is None:
            return img, []

        # Refresh embeddings for the specific user
        if user_id is not None:
            self._refresh_embeddings(user_id=user_id)
        else:
            self._refresh_embeddings()  # Fallback to all users if no user_id provided

        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face_roi = img_rgb[y1:y2, x1:x2]
            try:
                aligned = self.fr.process_image(face_roi)
                if aligned is None:
                    continue
                embedding = self.fr.generate_embeddings(aligned)
                embedding /= np.linalg.norm(embedding)
                if self.index is None:
                    print("No embeddings available for recognition.")
                    continue

                _, indices = self.index.search(embedding.reshape(1, -1), 3)
                votes = {}
                for idx in indices[0]:
                    label = self.labels[idx]
                    votes[label] = votes.get(label, 0) + 1
                if votes:
                    best_match = max(votes, key=votes.get)
                    confidence = votes[best_match] / 3
                    matched_user_id = self.user_ids[
                        self.labels.tolist().index(best_match)
                    ]

                    # Only include match if it corresponds to the provided user_id
                    if user_id is None or matched_user_id == user_id:
                        matches.append(
                            {
                                "user_id": int(matched_user_id),
                                "name": best_match,
                                "confidence": float(confidence),
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                            }
                        )
                    else:
                        print(
                            f"Match found for user {matched_user_id}, but expected {user_id}"
                        )

                    color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_rgb,
                        f"{best_match} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue
        output_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # If no matches and faces were detected, return "unknown"
        if not matches and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                matches.append(
                    {
                        "user_id": 0,
                        "name": "unknown",
                        "confidence": 0.5,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )
        return output_img, matches


# Define the exact model used during training
class PoseAwareResNet(torch.nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        num_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 3),
        )
        self.pose_classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, 256), torch.nn.ReLU(), torch.nn.Linear(256, 5)
        )  # Five classes: frontal, left, right, up, down

    def forward(self, x):
        features = self.backbone(x)
        angles = self.regressor(features)
        pose_logits = self.pose_classifier(features)
        return angles, pose_logits


# Instantiate model and load weights
model = PoseAwareResNet().to(device)

# Load checkpoint
checkpoint = torch.load(head_pose_model_path, map_location=device)
model_state_dict = checkpoint.get("model_state_dict", checkpoint)

# Load weights
model.load_state_dict(model_state_dict)
model.eval()

# Define normalization parameters
angle_ranges = {"yaw": (-75, 75), "pitch": (-60, 80), "roll": (-80, 40)}
normalization_factors = {
    "yaw": 1.0 / max(abs(angle_ranges["yaw"][0]), abs(angle_ranges["yaw"][1])),
    "pitch": 1.0 / max(abs(angle_ranges["pitch"][0]), abs(angle_ranges["pitch"][1])),
    "roll": 1.0 / max(abs(angle_ranges["roll"][0]), abs(angle_ranges["roll"][1])),
}


def denormalize_angle(normalized_angle, angle_type):
    """Convert normalized angle back to raw degrees"""
    return normalized_angle / normalization_factors[angle_type]


# Preprocessing transform
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Pose thresholds
POSE_THRESHOLDS = {
    "frontal": (-10, 10),
    "left": (20, float("inf")),
    "right": (-float("inf"), -15),
    "up": (20, float("inf")),
    "down": (-float("inf"), -15),
}


def get_pose_category(pitch, yaw):
    """Determine pose category based on thresholds"""
    if (
        POSE_THRESHOLDS["frontal"][0] <= yaw <= POSE_THRESHOLDS["frontal"][1]
        and POSE_THRESHOLDS["frontal"][0] <= pitch <= POSE_THRESHOLDS["frontal"][1]
    ):
        return "frontal"
    elif yaw < POSE_THRESHOLDS["right"][1]:
        return "right"
    elif yaw > POSE_THRESHOLDS["left"][0]:
        return "left"
    elif pitch > POSE_THRESHOLDS["up"][0]:
        return "up"
    elif pitch < POSE_THRESHOLDS["down"][1]:
        return "down"
    return "frontal"


# Exponential Moving Average (EMA) for smoothing
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ema_yaw = None
        self.ema_pitch = None
        self.ema_roll = None

    def update(self, yaw, pitch, roll):
        if self.ema_yaw is None:
            self.ema_yaw = yaw
            self.ema_pitch = pitch
            self.ema_roll = roll
        else:
            self.ema_yaw = self.alpha * yaw + (1 - self.alpha) * self.ema_yaw
            self.ema_pitch = self.alpha * pitch + (1 - self.alpha) * self.ema_pitch
            self.ema_roll = self.alpha * roll + (1 - self.alpha) * self.ema_roll
        return self.ema_yaw, self.ema_pitch, self.ema_roll


# Initialize EMA smoother
ema_smoother = EMA(alpha=0.3)

# Cheating score weights
CHEATING_WEIGHTS = {
    "multiple_faces": 30,
    "non_frontal_pose": 10,
    "suspicious_object": 20,
}


# Function to detect faces using YOLO
def detect_faces(image: Image.Image) -> List[Dict]:
    """Run face detection"""
    try:
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # Enhance image: increase brightness and contrast
        image_cv = cv2.convertScaleAbs(image_cv, alpha=1.5, beta=20)
        # Optional: Resize to match training resolution
        image_cv = cv2.resize(image_cv, (640, 640))
        results = face_detector(image_cv)
        print(f"YOLO raw results: {results}")
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Scale bounding box back to original image size
                    orig_width, orig_height = image.size
                    x1 = int(x1 * orig_width / 640)
                    y1 = int(y1 * orig_height / 640)
                    x2 = int(x2 * orig_width / 640)
                    y2 = int(y2 * orig_height / 640)
                    faces.append(
                        {
                            "bounding_box": [x1, y1, x2, y2],
                            "confidence": float(box.conf),
                        }
                    )
        print(f"Detected faces: {faces}")
        return faces
    except Exception as e:
        print(f"Error in detect_faces: {str(e)}")
        return []


# Function to process a single image and detect poses
async def detect_head_pose(face_region: Image.Image):
    """Run head pose estimation"""
    input_tensor = transform(face_region).unsqueeze(0).to(device)
    with torch.no_grad():
        angles, _ = model(input_tensor)
        pitch, yaw, roll = angles[0].cpu().numpy()

    # Denormalize
    pitch = denormalize_angle(pitch, "pitch")
    yaw = denormalize_angle(yaw, "yaw")
    roll = denormalize_angle(roll, "roll")

    # Apply EMA smoothing
    smoothed_yaw, smoothed_pitch, smoothed_roll = ema_smoother.update(yaw, pitch, roll)

    # Get pose category
    pose_category = get_pose_category(smoothed_pitch, smoothed_yaw)

    return {
        "pose": pose_category,
        "yaw": float(smoothed_yaw),
        "pitch": float(smoothed_pitch),
        "roll": float(smoothed_roll),
    }


async def process_image(image: Image.Image) -> Dict:
    """Process image with all models"""
    try:
        # Run face detection (not async)
        faces = detect_faces(image)

        # Process head pose for each detected face
        head_poses = []
        for face in faces:
            x1, y1, x2, y2 = face["bounding_box"]
            face_region = image.crop((x1, y1, x2, y2))
            head_pose = await detect_head_pose(face_region)
            head_poses.append(head_pose)

        # Compute cheating score
        score_increment = 0
        alerts = []

        # Check for multiple faces
        if len(faces) > 1:
            score_increment += CHEATING_WEIGHTS["multiple_faces"]
            alerts.append("Multiple faces detected")
        elif len(faces) == 0:
            alerts.append("No faces detected")

        # Check for non-frontal pose
        non_frontal_poses = [pose for pose in head_poses if pose["pose"] != "frontal"]
        if non_frontal_poses:
            score_increment += CHEATING_WEIGHTS["non_frontal_pose"]
            alerts.append("Non-frontal pose detected")

        return {
            "faces": faces,
            "head_poses": head_poses,
            "score_increment": score_increment,
            "alerts": alerts,
        }
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return {"faces": [], "head_poses": [], "alerts": [], "score_increment": 0}


class RegisterRequest(BaseModel):
    user_id: int
    images: List[str]


app = FastAPI(title="Face Recognition API")

# Store WebSocket connections by student_id and quiz_id
connections: Dict[str, WebSocket] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_db = FaceDB()
fr = FaceRecognition()
yolo = YOLO(FACE_MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
recognizer = RealTimeRecognizer()


@app.post("/register")
async def register_student(data: RegisterRequest):
    user_id = data.user_id
    images = data.images
    print("Received:", {"user_id": user_id, "images": images[:3]})
    try:
        # First verify the user exists before processing
        conn = mysql.connector.connect(
            host=DB_HOST,
            database=DB_DATABASE,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            port=int(DB_PORT),
        )
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        user_result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user_result:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"User with ID {user_id} does not exist in the database",
                },
                status_code=404,
            )

        if len(images) < 3:
            raise HTTPException(400, "At least 3 images required")

        embeddings = []
        processor = BatchProcessor()

        for img_b64 in images:
            try:
                if "," in img_b64:
                    img_b64 = img_b64.split(",")[1]

                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                aligned = processor.fr.process_image(img)
                if aligned is None:
                    continue

                embedding = processor.fr.generate_embeddings(aligned)
                embeddings.append(embedding)

            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

        if embeddings:
            processor.db.store_embeddings(user_id, embeddings)
            recognizer._refresh_embeddings()

            return JSONResponse(
                {
                    "status": "success",
                    "message": f"Registered {len(embeddings)} embeddings for user {user_id}",
                }
            )
        else:
            raise HTTPException(400, "No valid faces detected")

    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/recognize")
async def recognize_face(request_data: dict):
    try:
        captured_image = request_data.get("captured_image")
        user_id = request_data.get("user_id")
        if not captured_image:
            raise HTTPException(400, "Invalid image")

        # Extract base64 data (remove "data:image/jpeg;base64," prefix if present)
        if "," in captured_image:
            captured_image = captured_image.split(",")[1]

        img_data = base64.b64decode(captured_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(400, "Invalid image")

        processed_img, matches = recognizer.process_image(img, user_id=user_id)

        print(f"Recognition results for user_id {user_id}: {matches}")

        # Ensure matches always contains at least detection info, even if no recognition
        yolo_results = yolo(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if (
            not matches and yolo_results[0].boxes is not None
        ):  # If YOLO detected faces but no recognition
            for box in yolo_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                matches.append(
                    {
                        "user_id": 0,
                        "name": "unknown",
                        "confidence": 0.5,  # Default confidence for detected but unrecognized faces
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

        return {
            "status": "success",
            "matches": matches,
            "detected_faces": len(matches) > 0,
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/students")
async def list_students():
    try:
        names = face_db.get_student_names()
        return JSONResponse({"status": "success", "students": names})
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# Pydantic model for /process_periodic input
class ProcessPeriodicRequest(BaseModel):
    student_id: str
    quiz_id: str
    image_b64: str
    auth_token: str


# Pydantic model for /submit_due_to_cheating input
class SubmitDueToCheatingRequest(BaseModel):
    student_id: str
    quiz_id: str
    answers: List[Dict[str, str]]  # List of {question_id, answer}
    auth_token: str


@app.websocket("/ws/{student_id}/{quiz_id}")
async def websocket_endpoint(websocket: WebSocket, student_id: str, quiz_id: str):
    await websocket.accept()
    session_key = f"{student_id}_{quiz_id}"
    connections[session_key] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket message: {data}")  # Log incoming messages
    except WebSocketDisconnect:
        if session_key in connections:
            del connections[session_key]
    finally:
        if session_key in connections:
            del connections[session_key]


async def notify_client(student_id: str, quiz_id: str, message: dict):
    """Send notification to client via WebSocket"""
    session_key = f"{student_id}_{quiz_id}"
    if session_key in connections:
        try:
            await connections[session_key].send_json(message)
        except Exception as e:
            print(f"Error sending WebSocket message: {str(e)}")


async def log_cheating_to_laravel(
    student_id: str,
    quiz_id: str,
    alerts: List[str],
    score_increment: int,
    auth_token: str,
):
    """Send suspicious behaviors and score update to Laravel"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/quizzes/update-cheating-score",
                json={
                    "student_id": student_id,
                    "quiz_id": quiz_id,
                    "score_increment": score_increment,
                    "alerts": alerts,
                },
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Accept": "application/json",
                },
                timeout=10.0,  # Add timeout to prevent hanging
            )
            response_data = response.json()
            if response.status_code != 200:
                print(
                    f"Failed to log to Laravel: {response.text} (Status: {response.status_code})"
                )
            else:
                print(f"Successfully logged to Laravel: {response.text}")
            return response_data
        except Exception as e:
            print(f"Error logging to Laravel: {str(e)}")
            return {"message": "Error logging to Laravel", "error": str(e)}


async def submit_to_laravel(
    student_id: str, quiz_id: str, answers: List[Dict[str, str]], auth_token: str
):
    """Submit quiz answers to Laravel when cheating score reaches 100"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"http://localhost:8000/api/quizzes/submit/{quiz_id}",
                json={
                    "student_id": student_id,
                    "answers": answers,
                },
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Accept": "application/json",
                },
                timeout=10.0,
            )
            return {"status": response.status_code, "data": response.json()}
        except Exception as e:
            print(f"Error submitting to Laravel: {str(e)}")
            return {
                "status": 500,
                "data": {"message": "Error submitting quiz", "error": str(e)},
            }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    student_id: str = None,
    quiz_id: str = None,
    auth_token: str = None,
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = await process_image(image)
        if result["alerts"] and student_id and quiz_id and auth_token:
            laravel_response = await log_cheating_to_laravel(
                student_id,
                quiz_id,
                result["alerts"],
                result["score_increment"],
                auth_token,
            )
            ws_message = {
                "type": "alert",
                "message": result["alerts"],
                "score_increment": result["score_increment"],
                "auto_submitted": laravel_response.get("auto_submitted", False),
                "new_score": laravel_response.get("new_score", 0),
            }
            await notify_client(student_id, quiz_id, ws_message)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/process_periodic")
async def process_periodic(request: ProcessPeriodicRequest):
    print(
        f"Received: student_id={request.student_id}, quiz_id={request.quiz_id}, image_b64_length={len(request.image_b64)}, auth_token={request.auth_token[:10]}..."
    )
    try:
        # Decode base64 string
        image_b64 = request.image_b64
        if "," in image_b64:
            image_data = base64.b64decode(image_b64.split(",")[1])
        else:
            image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        result = await process_image(image)
        if not isinstance(result, dict):
            raise ValueError("process_image must return a dictionary")
        if "alerts" not in result or "score_increment" not in result:
            raise ValueError(
                "process_image result missing 'alerts' or 'score_increment'"
            )
        if result["alerts"]:
            laravel_response = await log_cheating_to_laravel(
                request.student_id,
                request.quiz_id,
                result["alerts"],
                result["score_increment"],
                request.auth_token,
            )
            ws_message = {
                "type": "alert",
                "message": result["alerts"],
                "score_increment": result["score_increment"],
                "auto_submitted": laravel_response.get(
                    "auto_submitted", False
                ),  # Forward flag
                "new_score": laravel_response.get(
                    "new_score", laravel_response.get("score", 0)
                ),  # Use new_score or fallback to score
            }
            await notify_client(request.student_id, request.quiz_id, ws_message)
        return JSONResponse(content=result)
    except base64.binascii.Error:
        print("Error: Invalid base64 string")
        return JSONResponse(content={"error": "Invalid base64 string"}, status_code=422)
    except ValueError as ve:
        print(f"Error in process_periodic: {str(ve)}")
        return JSONResponse(content={"error": str(ve)}, status_code=500)
    except Exception as e:
        print(f"Error in process_periodic: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/submit_due_to_cheating")
async def submit_due_to_cheating(request: SubmitDueToCheatingRequest):
    try:
        response = await submit_to_laravel(
            request.student_id,
            request.quiz_id,
            request.answers,
            request.auth_token,
        )
        if response["status"] == 200:
            return JSONResponse(
                content={"status": 200, "message": "Quiz submitted due to cheating"}
            )
        else:
            return JSONResponse(
                content=response["data"], status_code=response["status"]
            )
    except Exception as e:
        print(f"Error in submit_due_to_cheating: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
