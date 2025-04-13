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

app = Flask(__name__)

# Global variables to share data between routes
authentication_data = {
    'status': 'ready',
    'output': [],
    'current_frame': None,
    'face_frame1': None,
    'face_frame2': None,
    'eye_frame': None,
    'matched_name': None,
    'liveness_face': False,
    'iris_match': False
}

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
        img = cv2.imread(iris_path, cv2.IMREAD_GRAYSCALE)
        return img
    return None

# -------------------------------
# ⚙️ CONFIGURATION
# -------------------------------
dataset_path = "D:/PythonPrograms/lfw-deepfunneled/lfw-deepfunneled"
selected_people = ['ABCD', 'ABCD2', 'ABCD3', 'ABCD4']

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
# Update the iris_authentication function
def iris_authentication():
    if not authentication_data['matched_name']:
        return
    
    authentication_data['output'].append("\n[INFO] Starting Iris Authentication...")
    
    # Wait for eye frame to be captured
    start_time = time.time()
    while authentication_data['eye_frame'] is None and (time.time() - start_time) < 10:
        time.sleep(0.1)
    
    if authentication_data['eye_frame'] is None:
        authentication_data['output'].append("[ERROR] No eye frame captured")
        return
    
    # Process captured eye frame
    eye_frame = authentication_data['eye_frame']
    
    # Iris comparison
    authentication_data['output'].append("[INFO] Comparing iris regions...")
    person_folder = os.path.join(dataset_path, authentication_data['matched_name'])
    stored_iris_gray = load_stored_iris(person_folder)

    try:
        rgb_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)
        iris_live = extract_eye_region(rgb_eye)

        if stored_iris_gray is not None and iris_live is not None:
            iris_live_gray = cv2.cvtColor(iris_live, cv2.COLOR_RGB2GRAY)
            iris_live_gray = preprocess_iris(iris_live_gray)
            iris_live_resized = cv2.resize(iris_live_gray, (stored_iris_gray.shape[1], stored_iris_gray.shape[0]))

            iris_score = compare_histograms(stored_iris_gray, iris_live_resized)
            authentication_data['output'].append(f"[DEBUG] Iris histogram correlation: {iris_score:.4f}")

            if iris_score >= 0.5:
                authentication_data['iris_match'] = True
                authentication_data['output'].append("[SUCCESS] Iris matched ✅")
            else:
                authentication_data['output'].append("[FAIL] Iris not matched ❌")
        else:
            authentication_data['output'].append("[FAIL] Could not extract iris region ❌")
    except Exception as e:
        authentication_data['output'].append(f"[ERROR] Iris processing failed: {str(e)}")

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
    if authentication_data['current_frame'] is not None:
        authentication_data['face_frame1'] = authentication_data['current_frame'].copy()
        authentication_data['output'].append("[INFO] First face frame captured.")
    return jsonify({'status': 'captured'})

@app.route('/capture_face2', methods=['POST'])
def capture_face2():
    if authentication_data['current_frame'] is not None:
        authentication_data['face_frame2'] = authentication_data['current_frame'].copy()
        authentication_data['output'].append("[INFO] Second face frame captured.")
    return jsonify({'status': 'captured'})

@app.route('/capture_eye', methods=['POST'])
def capture_eye():
    if authentication_data['current_frame'] is not None:
        authentication_data['eye_frame'] = authentication_data['current_frame'].copy()
        authentication_data['output'].append("[INFO] Eye frame captured.")
    return jsonify({'status': 'captured'})

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