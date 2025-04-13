# your_script.py
import cv2
import face_recognition
import numpy as np
import os
import random
import time
from skimage.metrics import structural_similarity as ssim

# Utility: Augment Face (Synthetic)
def augment_face(image):
    angle = random.choice([-10, -5, 5, 10])
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    flipped = cv2.flip(image, 1)
    return [rotated, flipped]

# Augment Encodings
def augment_encodings(image):
    encodings = []
    augmented = augment_face(image)
    for aug in augmented:
        enc = face_recognition.face_encodings(aug)
        if enc:
            encodings.append(enc[0])
    return encodings

# Extract Eye Region Function
def extract_eye_region(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(rgb_image)
    if landmarks:
        left_eye = landmarks[0].get("left_eye")
        if left_eye:
            xs = [pt[0] for pt in left_eye]
            ys = [pt[1] for pt in left_eye]
            x_min, x_max = max(0, min(xs) - 5), min(image.shape[1], max(xs) + 5)
            y_min, y_max = max(0, min(ys) - 5), min(image.shape[0], max(ys) + 5)
            return image[y_min:y_max, x_min:x_max]
    return None

# Preprocess Iris for Clarity
def preprocess_iris(image):
    image = cv2.equalizeHist(image)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

# Histogram Comparison Function
def compare_histograms(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# Load Stored Iris Image
def load_stored_iris(person_folder):
    iris_path = os.path.join(person_folder, "iris.jpg")
    if os.path.exists(iris_path):
        img = cv2.imread(iris_path, cv2.IMREAD_GRAYSCALE)
        return img
    return None

# Load Known Faces
def load_known_faces(dataset_path, selected_people):
    known_encodings = []
    known_names = []
    known_images = {}

    for person in selected_people:
        person_folder = os.path.join(dataset_path, person)
        if not os.path.isdir(person_folder): continue

        for file in os.listdir(person_folder):
            if file.endswith(".jpg") and file != "iris.jpg":
                img_path = os.path.join(person_folder, file)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(person)
                    known_images[person] = img

                    # Add few-shot synthetic embeddings
                    extra_encs = augment_encodings(img)
                    for enc in extra_encs:
                        known_encodings.append(enc)
                        known_names.append(person)

                break

    return known_encodings, known_names

# Face Liveness Detection
def detect_face_liveness(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    sim_index = ssim(gray1, gray2)
    return sim_index < 0.9, sim_index

# Face Recognition
def recognize_face(frame, known_encodings, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        face_dist = face_recognition.face_distance(known_encodings, encodings[0])
        best_idx = np.argmin(face_dist)
        if face_dist[best_idx] < 0.5:
            return known_names[best_idx]
    return None

# Iris Comparison
def compare_iris(iris_live, stored_iris_gray):
    iris_live_gray = cv2.cvtColor(iris_live, cv2.COLOR_RGB2GRAY)
    iris_live_gray = preprocess_iris(iris_live_gray)
    iris_live_resized = cv2.resize(iris_live_gray, (stored_iris_gray.shape[1], stored_iris_gray.shape[0]))
    iris_score = compare_histograms(stored_iris_gray, iris_live_resized)
    return iris_score >= 0.5, iris_score

# Full Authentication Pipeline
def full_authentication_pipeline():
    # Your camera and other setup logic here
    dataset_path = "D:/PythonPrograms/lfw-deepfunneled/lfw-deepfunneled"
    selected_people = ['ABCD', 'ABCD2', 'ABCD3', 'ABCD4']

    known_encodings, known_names = load_known_faces(dataset_path, selected_people)

    # Dummy frames to simulate this part of the code
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    liveness_face, sim_index = detect_face_liveness(frame1, frame2)
    
    if liveness_face:
        matched_name = recognize_face(frame2, known_encodings, known_names)
        iris_match = False
        if matched_name:
            iris_live = np.zeros((100, 100, 3), dtype=np.uint8)  # Example placeholder for live iris capture
            person_folder = os.path.join(dataset_path, matched_name)
            stored_iris_gray = load_stored_iris(person_folder)
            if stored_iris_gray is not None and iris_live is not None:
                iris_match, iris_score = compare_iris(iris_live, stored_iris_gray)
        
        if iris_match:
            return {"message": "Access Granted"}
        else:
            return {"message": "Iris Match Failed"}
    else:
        return {"message": "Face Liveness Failed"}
