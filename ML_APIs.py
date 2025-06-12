import os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "TRUE"  # Allows multiple OpenMP runtimes (temporary fix)
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
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
from pydantic import BaseModel, ValidationError
from io import BytesIO
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import io
import asyncio
import json
import httpx
import mediapipe as mp
from collections import deque
import torch.nn as nn
import asyncio

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH")
HEAD_POSE_MODEL_PATH = os.getenv("HEAD_POSE_MODEL_PATH")
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH")

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

# Load object detection model
object_detection_model = YOLO(OBJECT_DETECTION_MODEL_PATH).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
    "multiple_faces": 20,
    "non_frontal_pose": 5,
    "suspicious_object": 15,
    "suspicious_gaze": 5,
    "no_faces_detected": 10,
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


def detect_objects(image: Image.Image) -> List[Dict]:
    """Run object detection to check for suspicious objects"""
    try:
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # Resize to match training resolution (640x640)
        image_cv = cv2.resize(image_cv, (640, 640))
        results = object_detection_model(image_cv)
        print(f"YOLO raw results: {results}")

        suspicious_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf >= 0.5:
                    cls = int(box.cls[0])  # Class index
                    class_name = result.names[cls]  # Get class name
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Scale back to original image size
                    orig_width, orig_height = image.size
                    x1 = int(x1 * orig_width / 640)
                    y1 = int(y1 * orig_height / 640)
                    x2 = int(x2 * orig_width / 640)
                    y2 = int(y2 * orig_height / 640)
                    suspicious_objects.append(
                        {
                            "class": class_name,
                            "bounding_box": [x1, y1, x2, y2],
                            "confidence": float(box.conf),
                        }
                    )
        print(f"Suspicious objects: {suspicious_objects}")
        return suspicious_objects
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        return []


# Gaze tracking model definition
class GazeResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for i, (name, param) in enumerate(self.base_model.named_parameters()):
            if i < 30:
                param.requires_grad = False
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.base_model(x)


class GazeKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(3, 3)
        self.kf.measurementMatrix = np.eye(3, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(3, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(3, dtype=np.float32) * 0.003
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.03

    def update(self, measurement):
        self.kf.predict()
        return self.kf.correct(measurement.reshape(3, 1)).flatten()


def extract_eye_region(image, landmarks, eye_indices, expand_ratio=0.2, min_size=20):
    points = np.array(
        [
            (landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0])
            for i in eye_indices
        ]
    )
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, int(x_min - width * expand_ratio))
    y_min = max(0, int(y_min - height * expand_ratio))
    x_max = min(image.shape[1], int(x_max + width * expand_ratio))
    y_max = min(image.shape[0], int(y_max + height * expand_ratio))

    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def preprocess_eye(eye_img):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)))


def classify_gaze(gaze_vector, yaw_thresh=15):
    normalized = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)
    x, y, z = normalized

    yaw = np.degrees(np.arctan2(x, z + 1e-6))
    pitch = np.degrees(np.arcsin(y * 0.75))
    yaw = (yaw + 180) % 360 - 180
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180

    dynamic_scale = 1.2 - (abs(pitch) / 45)
    yaw_thresh *= dynamic_scale

    pitch_down_thresh = 6.0
    pitch_up_thresh = 11.0

    horizontal = (
        "Right" if yaw > yaw_thresh else "Left" if yaw < -yaw_thresh else "Center"
    )
    vertical = (
        "Down"
        if pitch > pitch_down_thresh
        else "Up" if pitch < -pitch_up_thresh else "Center"
    )

    if vertical != "Center":
        direction = (
            f"{vertical} {horizontal}" if horizontal != "Center" else f"{vertical}"
        )
    else:
        direction = f"{horizontal}"

    return direction.strip(), yaw, pitch


def detect_gaze(image: Image.Image) -> Dict:
    """Run gaze tracking on the image"""
    print("Starting gaze detection...")
    try:
        # Convert PIL Image to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_cv = cv2.resize(image_cv, (640, 640))
        image_cv = cv2.flip(image_cv, 1)
        print("Image converted for gaze detection.")

        # Face detection
        results = mp_face.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print("Gaze detection: No face detected by MediaPipe.")
            return {"status": "no_face_detected", "message": "No face detected"}

        # Face mesh for landmarks
        mesh_results = mp_face_mesh.process(image_cv)
        if not mesh_results.multi_face_landmarks:
            print("Gaze detection: Face detected but no landmarks found.")
            return {
                "status": "no_face_landmarks",
                "message": "Face detected but no landmarks",
            }

        face_landmarks = mesh_results.multi_face_landmarks[0]
        print("Gaze detection: Face landmarks detected.")

        # Process both eyes
        left_eye, left_box = extract_eye_region(
            image_cv,
            face_landmarks.landmark,
            [
                33,
                7,
                163,
                144,
                145,
                153,
                154,
                155,
                133,
                173,
                157,
                158,
                159,
                160,
                161,
                246,
            ],
        )
        right_eye, right_box = extract_eye_region(
            image_cv,
            face_landmarks.landmark,
            [
                362,
                382,
                381,
                380,
                374,
                373,
                390,
                249,
                263,
                466,
                388,
                387,
                386,
                385,
                384,
                398,
            ],
        )
        print("Gaze detection: Eye regions extracted.")

        # Get gaze predictions
        with torch.no_grad():
            left_input = preprocess_eye(left_eye).unsqueeze(0).to(device)
            right_input = preprocess_eye(right_eye).unsqueeze(0).to(device)
            left_gaze = gaze_model(left_input)[0].cpu().numpy()
            right_gaze = gaze_model(right_input)[0].cpu().numpy()
        print("Gaze detection: Gaze predictions obtained.")

        # Temporal smoothing
        current_gaze = (left_gaze + right_gaze) / 2
        gaze_history.append(current_gaze)
        smoothed_gaze = np.mean(gaze_history, axis=0)

        # Kalman filtering
        filtered_gaze = kf.update(smoothed_gaze)

        # Classify gaze direction
        direction, yaw, pitch = classify_gaze(filtered_gaze)
        print(
            f"Gaze detection successful: direction={direction}, yaw={yaw}, pitch={pitch}"
        )

        return {
            "status": "success",
            "gaze_direction": direction,
            "yaw_degrees": float(yaw),
            "pitch_degrees": float(pitch),
            "gaze_vector": filtered_gaze.tolist(),
        }
    except Exception as e:
        print(f"Error in detect_gaze: {str(e)}")
        return {"status": "error", "message": str(e)}


# Global dictionaries to track sequences of non-frontal pose and suspicious gaze detections
non_frontal_pose_sequences: Dict[str, deque] = {}
suspicious_gaze_sequences: Dict[str, deque] = {}


async def process_image(
    image: Image.Image, student_id: str = None, quiz_id: str = None
) -> Dict:
    """Process image with all models"""
    try:
        # Define async wrappers for non-async detection functions
        async def run_face_detection(img):
            return detect_faces(img)

        async def run_object_detection(img):
            return detect_objects(img)

        async def run_gaze_detection(img):
            result = detect_gaze(img)
            print(f"Gaze detection result: {result}")
            return result

        # Run all detections in parallel
        face_task = run_face_detection(image)
        head_pose_tasks = []
        object_task = run_object_detection(image)
        gaze_task = run_gaze_detection(image)

        # Await all tasks
        faces, suspicious_objects, gaze_result = await asyncio.gather(
            face_task, object_task, gaze_task
        )

        # Process head pose for each detected face
        for face in faces:
            x1, y1, x2, y2 = face["bounding_box"]
            face_region = image.crop((x1, y1, x2, y2))
            head_pose_tasks.append(detect_head_pose(face_region))

        head_poses = await asyncio.gather(*head_pose_tasks)

        # Compute cheating score
        score_increment = 0
        alerts = []

        # Generate session key if student_id and quiz_id are provided
        session_key = f"{student_id}_{quiz_id}" if student_id and quiz_id else None

        # Check for multiple faces
        if len(faces) > 1:
            score_increment += CHEATING_WEIGHTS["multiple_faces"]
            alerts.append("Multiple faces detected")
        elif len(faces) == 0:
            alerts.append("No faces detected")
            score_increment += CHEATING_WEIGHTS["no_faces_detected"]

        # Initialize sequences if not present
        if session_key and session_key not in non_frontal_pose_sequences:
            non_frontal_pose_sequences[session_key] = deque(maxlen=3)
        if session_key and session_key not in suspicious_gaze_sequences:
            suspicious_gaze_sequences[session_key] = deque(maxlen=3)

        # Check for non-frontal pose
        non_frontal_poses = [pose for pose in head_poses if pose["pose"] != "frontal"]
        if non_frontal_poses and session_key:
            alerts.append("Non-frontal pose detected")
            # Add True to the sequence for non-frontal pose
            non_frontal_pose_sequences[session_key].append(True)
            if len(non_frontal_pose_sequences[session_key]) == 3 and all(
                non_frontal_pose_sequences[session_key]
            ):
                score_increment += CHEATING_WEIGHTS["non_frontal_pose"]
                alerts.append("Non-frontal pose detected (3rd consecutive occurrence)")
                # Clear the sequence after incrementing the score
                non_frontal_pose_sequences[session_key].clear()
        elif non_frontal_poses:
            alerts.append("Non-frontal pose detected")
        elif session_key:
            # Add False to the sequence for frontal pose
            non_frontal_pose_sequences[session_key].append(False)

        # Check for suspicious objects
        if suspicious_objects:
            score_increment += CHEATING_WEIGHTS["suspicious_object"]
            for obj in suspicious_objects:
                alerts.append(f"Suspicious object detected: {obj['class']}")

        # Check for suspicious gaze direction
        if gaze_result["status"] == "success":
            gaze_direction = gaze_result["gaze_direction"]
            print(f"Processing gaze direction: {gaze_direction}")
            if gaze_direction not in ["Center", "Up", "Down"] and session_key:
                alerts.append(f"Suspicious gaze direction: {gaze_direction}")
                # Add True to the sequence for suspicious gaze
                suspicious_gaze_sequences[session_key].append(True)
                if len(suspicious_gaze_sequences[session_key]) == 3 and all(
                    suspicious_gaze_sequences[session_key]
                ):
                    score_increment += CHEATING_WEIGHTS["suspicious_gaze"]
                    alerts.append(
                        "Suspicious gaze threshold reached (3rd consecutive occurrence)"
                    )
                    # Clear the sequence after incrementing the score
                    suspicious_gaze_sequences[session_key].clear()
            elif session_key:
                # Add False to the sequence for non-suspicious gaze
                suspicious_gaze_sequences[session_key].append(False)
        elif gaze_result["status"] == "error":
            print(f"Gaze detection failed: {gaze_result['message']}")
        else:
            print(
                f"Gaze detection status: {gaze_result['status']}, message: {gaze_result['message']}"
            )

        return {
            "faces": faces,
            "head_poses": head_poses,
            "suspicious_objects": suspicious_objects,
            "gaze_result": gaze_result,
            "score_increment": score_increment,
            "alerts": alerts,
        }
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return {
            "faces": [],
            "head_poses": [],
            "suspicious_objects": [],
            "gaze_result": {"status": "error", "message": str(e)},
            "alerts": [],
            "score_increment": 0,
        }


class RegisterRequest(BaseModel):
    user_id: int
    images: List[str]


face_db = FaceDB()
fr = FaceRecognition()
yolo = YOLO(FACE_MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
recognizer = RealTimeRecognizer()

# New global variables for gaze tracking
CHECKPOINT_DIR = os.getenv("GAZE_ESTIMATION_MODEL_PATH", "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_model = None
mp_face = None
mp_face_mesh = None
gaze_history = deque(maxlen=5)
kf = GazeKalmanFilter()

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gaze_model, mp_face, mp_face_mesh, yolo_face, object_detector
    try:
        print("Loading YOLO face model...")
        yolo_face = YOLO(FACE_MODEL_PATH).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("YOLO face model loaded.")

        print("Loading YOLO object detection model...")
        object_detector = YOLO(os.getenv("OBJECT_DETECTION_MODEL_PATH")).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print("YOLO object detection model loaded.")

        print("Loading gaze model...")
        gaze_model = GazeResNet18()
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_gaze_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        gaze_model.load_state_dict(state_dict)
        gaze_model.eval()
        gaze_model.to(device)
        print("Gaze model loaded.")

        print("Loading MediaPipe face detection...")
        mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        print("MediaPipe face detection loaded.")

        print("Loading MediaPipe face mesh...")
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_tracking_confidence=0.3,
        )
        print("MediaPipe face mesh loaded.")
    except Exception as e:
        print(f"Failed to load models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    yield
    print("Shutting down application...")


app = FastAPI(title="Face Recognition API", lifespan=lifespan)

# Store WebSocket connections by student_id and quiz_id
connections: Dict[str, WebSocket] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class Answer(BaseModel):
    question_id: int
    answer: str


# Pydantic model for /process_periodic input
class ProcessPeriodicRequest(BaseModel):
    student_id: str
    quiz_id: str
    image_b64: str
    auth_token: str
    image_b64: str
    answers: List[Answer] = []


# Pydantic model for /submit_due_to_cheating input
class SubmitDueToCheatingRequest(BaseModel):
    student_id: str
    quiz_id: str
    answers: List[Answer]
    auth_token: str


@app.websocket("/ws/{student_id}/{quiz_id}")
async def websocket_endpoint(websocket: WebSocket, student_id: str, quiz_id: str):
    await websocket.accept()
    session_key = f"{student_id}_{quiz_id}"
    connections[session_key] = websocket
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket message: {data}")
    except WebSocketDisconnect:
        if session_key in connections:
            del connections[session_key]
        if session_key in non_frontal_pose_sequences:
            del non_frontal_pose_sequences[session_key]
        if session_key in suspicious_gaze_sequences:
            del suspicious_gaze_sequences[session_key]
    finally:
        if session_key in connections:
            del connections[session_key]
        if session_key in non_frontal_pose_sequences:
            del non_frontal_pose_sequences[session_key]
        if session_key in suspicious_gaze_sequences:
            del suspicious_gaze_sequences[session_key]


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
    image_b64: str = None,
    answers: List[Dict[str, str]] = None,
):
    """Send suspicious behaviors and score update to Laravel, and cheating image to Laravel"""
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "student_id": student_id,
                "quiz_id": quiz_id,
                "score_increment": score_increment,
                "alerts": alerts,
            }
            if image_b64:  # Include image data if provided
                payload["image_b64"] = image_b64
            if answers:
                payload["answers"] = answers

            response = await client.post(
                "http://localhost:8000/api/quizzes/update-cheating-score",
                json=payload,
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
            # Ensure answers is a list of plain dictionaries
            serialized_answers = [
                (
                    {"question_id": answer["question_id"], "answer": answer["answer"]}
                    if isinstance(answer, dict)
                    else {"question_id": answer.question_id, "answer": answer.answer}
                )
                for answer in answers
            ]
            print(f"Sending answers to Laravel: {serialized_answers}")
            response = await client.post(
                f"http://localhost:8000/api/quizzes/submit/{quiz_id}",
                json={
                    "student_id": student_id,
                    "answers": serialized_answers,
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
        result = await process_image(image, request.student_id, request.quiz_id)
        if not isinstance(result, dict):
            raise ValueError("process_image must return a dictionary")
        if "alerts" not in result or "score_increment" not in result:
            raise ValueError(
                "process_image result missing 'alerts' or 'score_increment'"
            )
        if result["alerts"]:
            # Convert answers to plain dictionaries
            serialized_answers = (
                [
                    {"question_id": answer.question_id, "answer": answer.answer}
                    for answer in request.answers
                ]
                if hasattr(request, "answers")
                else []
            )
            laravel_response = await log_cheating_to_laravel(
                request.student_id,
                request.quiz_id,
                result["alerts"],
                result["score_increment"],
                request.auth_token,
                request.image_b64,
                serialized_answers,
            )
            ws_message = {
                "type": "alert",
                "message": result["alerts"],
                "score_increment": result["score_increment"],
                "auto_submitted": laravel_response.get("auto_submitted", False),
                "new_score": laravel_response.get(
                    "new_score", laravel_response.get("score", 0)
                ),
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
        # Convert Pydantic Answer objects to plain dictionaries
        serialized_answers = [
            {"question_id": answer.question_id, "answer": answer.answer}
            for answer in request.answers
        ]
        response = await submit_to_laravel(
            request.student_id,
            request.quiz_id,
            serialized_answers,
            request.auth_token,
        )
        if response["status"] == 200:
            return JSONResponse(
                content={"status": 200, "message": "Quiz submitted due to cheating"}
            )
        else:
            print(f"Failed to submit to Laravel: {response['data']}")
            return JSONResponse(
                content={
                    "error": "Failed to submit quiz to Laravel",
                    "details": response["data"],
                },
                status_code=response["status"],
            )
    except ValidationError as ve:
        print(f"Validation error in submit_due_to_cheating: {ve.errors()}")
        raise HTTPException(status_code=422, detail=ve.errors())
    except Exception as e:
        print(f"Error in submit_due_to_cheating: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
