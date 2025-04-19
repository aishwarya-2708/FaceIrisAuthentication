#runs successfully 1st app.py + added saving captured images of face and eyes and less strict iris authentication.
#previous which is final but not saving images and strict eye is sent through mail to mana.

from flask import Flask, render_template, Response, request, jsonify
import cv2
import face_recognition
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import random
import base64
import threading
from datetime import datetime

app = Flask(__name__)

# Global variables to share data between routes
authentication_data = {
    'status': 'ready',
    'output': [],
    'current_frame': None,
    'face_frame1': None,
    'face_frame2': None,
    'eye_frame': None,
    'iris_frame': None,
    'matched_name': None,
    'liveness_face': False,
    'iris_match': False
}


CAPTURES_DIR = "D:/PythonPrograms/face_iris_auth/captured_images"
FACE_DIR = os.path.join(CAPTURES_DIR, "faces")
IRIS_DIR = os.path.join(CAPTURES_DIR, "iris")

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(IRIS_DIR, exist_ok=True)

def save_captured_image(image, image_type, subfolder=""):
    """Save captured image with timestamp in specified subfolder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = os.path.join(CAPTURES_DIR, subfolder) if subfolder else CAPTURES_DIR
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{image_type}_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    return filename

def extract_iris_region(frame):
    """Enhanced iris extraction with better error handling"""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = face_recognition.face_landmarks(rgb)
        
        if not landmarks:
            return None
            
        # Get both eyes for better iris capture
        eyes = []
        for eye_type in ['left_eye', 'right_eye']:
            if eye_type in landmarks[0]:
                eye_points = landmarks[0][eye_type]
                xs = [p[0] for p in eye_points]
                ys = [p[1] for p in eye_points]
                x_min, x_max = max(0, min(xs)-10), min(frame.shape[1], max(xs)+10)
                y_min, y_max = max(0, min(ys)-10), min(frame.shape[0], max(ys)+10)
                eye = frame[y_min:y_max, x_min:x_max]
                eyes.append(eye)
        
        return eyes if eyes else None
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Iris extraction failed: {str(e)}")
        return None

# -------------------------------------
# ✨ Utility: Augment Face (Synthetic)
# -------------------------------------
def augment_face(image):
    angle = random.choice([-10, -5, 5, 10])
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    flipped = cv2.flip(image, 1)
    return [rotated, flipped]

# -------------------------------------
# ✨ Few-shot Learning Embedding Booster
# -------------------------------------
def augment_encodings(image):
    encodings = []
    augmented = augment_face(image)
    for aug in augmented:
        enc = face_recognition.face_encodings(aug)
        if enc:
            encodings.append(enc[0])
    return encodings

# -------------------------------
# 🧠 Extract Eye Region Function
# -------------------------------
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

# -------------------------------
# ✨ Preprocess Iris for Clarity
# -------------------------------
def preprocess_iris(image):
    image = cv2.equalizeHist(image)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image

# -------------------------------
# 📊 Histogram Comparison Function
# -------------------------------
def compare_histograms(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

# -------------------------------
# 📥 Load Stored Iris Image
# -------------------------------
def load_stored_iris(person_folder):
    iris_path = os.path.join(person_folder, "iris.jpg")
    if os.path.exists(iris_path):
        img = cv2.imread(iris_path)  # Keep as BGR (don't force grayscale)
        return img
    return None

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
dataset_path = "D:/PythonPrograms/lfw-deepfunneled/lfw-deepfunneled"
selected_people = ['ABCD', 'ABCD2', 'ABCD3', 'ABCD4','ABCD5']

# -------------------------------
# 🔍 Load Known Faces
# -------------------------------
def load_known_faces():
    authentication_data['output'].append("[INFO] Loading selected known faces with few-shot augmentation...")
    
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

                    authentication_data['output'].append(f"[LOADED] {person} with {1 + len(extra_encs)} encodings")
                break

    authentication_data['output'].append(f"[DONE] Total known encodings after few-shot learning: {len(known_encodings)}\n")
    return known_encodings, known_names, known_images

# -------------------------------
# 🎥 Face Authentication Process
# -------------------------------
def face_authentication():
    cap = cv2.VideoCapture(0)
    authentication_data['output'].append("[INFO] Face Camera Started... Waiting for frames...")
    
    # Wait for first frame
    while authentication_data['face_frame1'] is None:
        ret, frame = cap.read()
        if ret:
            authentication_data['current_frame'] = frame.copy()
        time.sleep(0.1)
    
    # Small delay between captures
    time.sleep(1)
    
    # Wait for second frame
    while authentication_data['face_frame2'] is None:
        ret, frame = cap.read()
        if ret:
            authentication_data['current_frame'] = frame.copy()
        time.sleep(0.1)
    
    cap.release()
    
    # Process frames
    frame1 = authentication_data['face_frame1']
    frame2 = authentication_data['face_frame2']
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    sim_index = ssim(gray1, gray2)
    authentication_data['liveness_face'] = sim_index < 0.9
    authentication_data['output'].append(f"[DEBUG] Face Liveness similarity index: {sim_index:.4f}")

    # Face recognition
    if authentication_data['liveness_face']:
        known_encodings, known_names, _ = load_known_faces()
        rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            face_dist = face_recognition.face_distance(known_encodings, encodings[0])
            best_idx = np.argmin(face_dist)
            if face_dist[best_idx] < 0.5:
                authentication_data['matched_name'] = known_names[best_idx]
                authentication_data['output'].append(f"[SUCCESS] Face matched with {known_names[best_idx]} ✅ (distance={face_dist[best_idx]:.2f})")
            else:
                authentication_data['output'].append("[FAIL] Face not matched ❌")
        else:
            authentication_data['output'].append("[FAIL] No face detected ❌")
    else:
        authentication_data['output'].append("[FAIL] Face liveness failed ❌")

# -------------------------------
# 👁️ Iris Authentication Process
# -------------------------------

# Add these new functions to app.py

def detect_iris_liveness(eye_frame1, eye_frame2):
    """Detect natural eye movement between frames"""
    if eye_frame1 is None or eye_frame2 is None:
        return False
    
    gray1 = cv2.cvtColor(eye_frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(eye_frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity
    score, _ = ssim(gray1, gray2, full=True)
    return score < 0.8  # Lower score means more movement

def capture_optimal_iris():
    """Capture the best quality iris frame"""
    cap = cv2.VideoCapture(0)
    best_iris = None
    best_score = -1
    
    for _ in range(30):  # Check 30 frames
        ret, frame = cap.read()
        if ret:
            eye_region = extract_eye_region(frame)
            if eye_region is not None:
                # Score frame quality
                gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if score > best_score:
                    best_score = score
                    best_iris = eye_region
    
    cap.release()
    return best_iris

def enhance_iris_quality(iris_img):
    """Handle both RGB and grayscale input"""
    try:
        # Convert to grayscale if needed
        if len(iris_img.shape) == 3:
            iris_img = cv2.cvtColor(iris_img, cv2.COLOR_BGR2GRAY)
        
        # Standard size
        iris_img = cv2.resize(iris_img, (100, 100))
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        iris_img = clahe.apply(iris_img)
        
        # Gentle processing
        iris_img = cv2.GaussianBlur(iris_img, (3,3), 0)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        iris_img = cv2.filter2D(iris_img, -1, kernel)
        
        return iris_img
    except Exception as e:
        print(f"Enhancement error: {str(e)}")
        return iris_img

def compare_iris(stored_iris, live_iris):
    """Handle both RGB and grayscale images"""
    try:
        # Convert stored iris to grayscale if it's RGB
        if len(stored_iris.shape) == 3:
            stored_gray = cv2.cvtColor(stored_iris, cv2.COLOR_BGR2GRAY)
        else:
            stored_gray = stored_iris.copy()
        
        # Convert live iris to grayscale if it's RGB
        if len(live_iris.shape) == 3:
            live_gray = cv2.cvtColor(live_iris, cv2.COLOR_BGR2GRAY)
        else:
            live_gray = live_iris.copy()
        
        # Standardize sizes
        height, width = stored_gray.shape
        live_resized = cv2.resize(live_gray, (width, height))
        
        # Enhance both images
        stored_enhanced = enhance_iris_quality(stored_gray)
        live_enhanced = enhance_iris_quality(live_resized)
        
        # Compare using multiple methods
        hist_score = compare_histograms(stored_enhanced, live_enhanced)
        ssim_score = ssim(stored_enhanced, live_enhanced)
        
        return (hist_score + ssim_score) / 2  # Average score
        
    except Exception as e:
        print(f"Comparison error: {str(e)}")
        return -1


# Update the iris_authentication function
def iris_authentication():
    if not authentication_data['matched_name']:
        authentication_data['status'] = 'complete'
        return
    
    try:
        authentication_data['output'].append("\n[INFO] Starting Iris Authentication...")
        
        # Capture iris
        iris_frame = capture_optimal_iris()
        if iris_frame is None:
            authentication_data['output'].append("[ERROR] Iris capture failed")
            authentication_data['status'] = 'complete'
            return
            
        authentication_data['eye_frame'] = iris_frame
        authentication_data['output'].append("[INFO] Processing iris...")
        
        # Load stored iris (could be RGB)
        person_folder = os.path.join(dataset_path, authentication_data['matched_name'])
        stored_iris = load_stored_iris(person_folder)
        
        if stored_iris is None:
            authentication_data['output'].append("[ERROR] No iris template found")
            authentication_data['status'] = 'complete'
            return
            
        # Compare
        score = compare_iris(stored_iris, iris_frame)
        
        if score < 0:
            authentication_data['output'].append("[ERROR] Comparison failed")
            authentication_data['iris_match'] = False
        else:
            authentication_data['output'].append(f"[DEBUG] Match score: {score:.4f}")
            authentication_data['iris_match'] = score >= 0.35  # Slightly more lenient threshold
            authentication_data['output'].append("[SUCCESS] Iris matched ✅" if authentication_data['iris_match'] else "[FAIL] Iris not matched ❌")
            
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Iris processing failed: {str(e)}")
        authentication_data['iris_match'] = False
    
    # Final result
    authentication_data['output'].append("\n========== FINAL RESULT ==========")
    if authentication_data['liveness_face'] and authentication_data['matched_name'] and authentication_data['iris_match']:
        authentication_data['output'].append(f"[✅ ACCESS GRANTED] Welcome, {authentication_data['matched_name']}!")
    else:
        authentication_data['output'].append("[❌ ACCESS DENIED] Authentication failed.")
    
    authentication_data['status'] = 'complete'


# -------------------------------
# 🚀 Main Authentication Process
# -------------------------------
def run_authentication():
    authentication_data['status'] = 'loading'
    authentication_data['output'] = []
    
    # Load known faces
    load_known_faces()
    
    # Face authentication
    authentication_data['status'] = 'face_capture'
    face_authentication()
    
    # Iris authentication if face matched
    if authentication_data['matched_name']:
        authentication_data['status'] = 'iris_capture'
        iris_authentication()
    else:
        authentication_data['status'] = 'complete'
        authentication_data['output'].append("\n========== FINAL RESULT ==========")
        authentication_data['output'].append("[❌ ACCESS DENIED] Authentication failed.")

# -------------------------------
# 🖥️ Flask Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_authentication', methods=['POST'])
def start_authentication():
    authentication_data['status'] = 'ready'
    authentication_data['output'] = []
    authentication_data['face_frame1'] = None
    authentication_data['face_frame2'] = None
    authentication_data['eye_frame'] = None
    authentication_data['matched_name'] = None
    authentication_data['liveness_face'] = False
    authentication_data['iris_match'] = False
    
    # Start authentication in a separate thread
    thread = threading.Thread(target=run_authentication)
    thread.start()
    
    return jsonify({'status': 'started'})


@app.route('/capture_face1', methods=['POST'])
def capture_face1():
    try:
        if authentication_data['current_frame'] is None:
            raise ValueError("No frame available for capture")
            
        authentication_data['face_frame1'] = authentication_data['current_frame'].copy()
        filename = save_captured_image(authentication_data['face_frame1'], "face1", "faces")
        authentication_data['output'].append(f"[SUCCESS] First face frame saved: {filename}")
        return jsonify({'status': 'success', 'path': filename})
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Face1 capture failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/capture_face2', methods=['POST'])
def capture_face2():
    try:
        if authentication_data['current_frame'] is None:
            raise ValueError("No frame available for capture")
            
        authentication_data['face_frame2'] = authentication_data['current_frame'].copy()
        filename = save_captured_image(authentication_data['face_frame2'], "face2", "faces")
        authentication_data['output'].append(f"[SUCCESS] Second face frame saved: {filename}")
        return jsonify({'status': 'success', 'path': filename})
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Face2 capture failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/capture_eye', methods=['POST'])
def capture_eye():
    try:
        if authentication_data['current_frame'] is None:
            raise ValueError("No frame available for capture")
            
        # Save full eye frame
        authentication_data['eye_frame'] = authentication_data['current_frame'].copy()
        full_eye_path = save_captured_image(authentication_data['eye_frame'], "eye_full", "iris")
        authentication_data['output'].append(f"[SUCCESS] Full eye frame saved: {full_eye_path}")
        
        # Extract and save iris regions
        iris_regions = extract_iris_region(authentication_data['current_frame'])
        if iris_regions:
            for i, iris_img in enumerate(iris_regions):
                if iris_img is not None and iris_img.size > 0:
                    iris_path = save_captured_image(iris_img, f"iris_{i}", "iris")
                    authentication_data['output'].append(f"[SUCCESS] Iris region {i} saved: {iris_path}")
                    if i == 0:
                        authentication_data['iris_frame'] = iris_img.copy()
        else:
            authentication_data['output'].append("[WARNING] No iris regions detected")
            
        return jsonify({'status': 'success'})
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Eye capture failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Store the current frame for capture
            authentication_data['current_frame'] = frame.copy()
            
            # Convert frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    output_html = "<br>".join(authentication_data['output'])
    return jsonify({
        'status': authentication_data['status'],
        'output': output_html
    })

@app.route('/stop_video_feed')
def stop_video_feed():
    # This will be called when authentication is complete
    return Response(status=204)

if __name__ == '__main__':
    app.run(debug=True)
