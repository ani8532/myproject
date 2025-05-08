import os
import time
import datetime
import cv2
import numpy as np
from flask import Blueprint, request, Response, jsonify, current_app
import threading
import traceback
from werkzeug.utils import secure_filename
from flask_jwt_extended import jwt_required, get_jwt_identity
import face_recognition
from functools import wraps
import easyocr
import difflib
from app import db
from app.models.models import FaceData, VehiclePlate, FaceEncoding
from app.models.user import User
from app.utils.auth_utils import get_user_by_email, load_user_face_db
from app.utils.email_utils import send_email_with_snapshot
ocr_reader = easyocr.Reader(['en'], gpu=False)
surveillance = Blueprint('surveillance', __name__)
from fuzzywuzzy import fuzz 
# Global resources
camera = None
camera_lock = threading.Lock()
camera_thread = None
camera_thread_running = False
frame_buffer = None
frame_lock = threading.Lock()
motion_thread = None
motion_thread_running = False
face_thread = None
vehicle_thread = None
vehicle_thread_running = False

# Ensure directories
os.makedirs('face_data', exist_ok=True)
os.makedirs('vehicle_data', exist_ok=True)
os.makedirs('motion_alerts', exist_ok=True)
UPLOAD_FOLDER = 'vehicle_plates'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ---------------- Token Decorator ---------------- #
def token_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user = get_jwt_identity()
        return fn(current_user, *args, **kwargs)
    return wrapper

# ---------------- Start Surveillance ---------------- #
@surveillance.route('/start', methods=['POST'])
@token_required
def start_surveillance(current_user):
    global camera, camera_thread, camera_thread_running
    global motion_thread, motion_thread_running, face_thread, vehicle_thread, vehicle_thread_running

    data = request.get_json()
    source = data.get('source', 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if camera and camera.isOpened():
        with camera_lock:
            camera.release()

    camera = cv2.VideoCapture(source)

    if not camera.isOpened():
        return jsonify({'msg': 'Failed to open camera'}), 500

    camera_thread_running = True
    if camera_thread is None or not camera_thread.is_alive():
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()

    motion_thread_running = True
    if motion_thread is None or not motion_thread.is_alive():
        motion_thread = threading.Thread(
            target=motion_detection_loop,
            args=(current_app._get_current_object(), current_user),
            daemon=True
        )
        motion_thread.start()

    if face_thread is None or not face_thread.is_alive():
        face_thread = threading.Thread(
            target=face_detection_loop,
            args=(current_app._get_current_object(), current_user),
            daemon=True
        )
        face_thread.start()
    
    vehicle_thread_running = True
    if vehicle_thread is None or not vehicle_thread.is_alive():
        vehicle_thread = threading.Thread(
            target=vehicle_detection_loop,
            args=(current_app._get_current_object(), current_user),
            daemon=True
        )
        vehicle_thread.start()

    return jsonify({'msg': 'Surveillance started'})

# ---------------- Stop Surveillance ---------------- #
@surveillance.route('/stop', methods=['POST'])
@token_required
def stop_surveillance(current_user):
    global motion_thread_running, camera_thread_running, camera_thread, vehicle_thread_running

    motion_thread_running = False
    camera_thread_running = False
    vehicle_thread_running = False
    time.sleep(1)

    if camera and camera.isOpened():
        with camera_lock:
            camera.release()

    camera_thread = None

    return jsonify({'msg': 'Surveillance stopped'}), 200

# ---------------- Camera Loop ---------------- #
def camera_loop():
    global camera, frame_buffer, camera_thread_running
    while camera_thread_running:
        try:
            with camera_lock:
                if camera and camera.isOpened():
                    success, frame = camera.read()
                    if success:
                        with frame_lock:
                            frame = cv2.resize(frame, (640, 480))
                            frame_buffer = frame.copy()
            time.sleep(0.01)
        except Exception as e:
            print("💥 Camera loop error:", e)
            traceback.print_exc()

# ---------------- Video Feed ---------------- #
@surveillance.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global frame_buffer
    while True:
        with frame_lock:
            local_frame = frame_buffer.copy() if frame_buffer is not None else None
        if local_frame is None or local_frame.size == 0:
            continue
        ret, buffer = cv2.imencode('.jpg', local_frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- Add Face ---------------- #
@surveillance.route('/add_face', methods=['POST'])
@token_required
def add_face(current_user):
    name = request.form.get('name')
    image = request.files.get('image')

    if not name or not image:
        return jsonify({'msg': 'Name and image are required'}), 400

    user = User.query.filter_by(email=current_user).first()
    filename = secure_filename(image.filename)
    path = os.path.join('face_data', filename)
    image.save(path)

    image_array = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image_array, model='small')

    if not encodings:
        return jsonify({'msg': 'No face found in the image.'}), 400

    encoding_bytes = encodings[0].tobytes()

    face_encoding = FaceEncoding(
        name=name,
        encoding=encoding_bytes,
        user_id=user.id if user else None,
        image_path=path
    )

    db.session.add(face_encoding)
    db.session.commit()

    return jsonify({"msg": "Face data saved successfully!"}), 201

# ---------------- Load Known Faces ---------------- #
def load_known_faces(db_faces):
    known_embeddings = []
    known_names = []
    for name, img_path, encoding in db_faces:
        if encoding is not None:
            known_embeddings.append(encoding)
            known_names.append(name)
    return known_embeddings, known_names

# ---------------- Motion Detection ---------------- #
def motion_detection_loop(app, user_email):
    global motion_thread_running, frame_buffer
    previous_frame = None
    last_motion_alert_time = 0
    motion_alert_interval = 60

    roi_top, roi_left, roi_bottom, roi_right = 100, 100, 400, 400

    with app.app_context():
        user = get_user_by_email(user_email)

    while motion_thread_running:
        try:
            with frame_lock:
                local_frame = frame_buffer.copy() if frame_buffer is not None else None
            if local_frame is None or local_frame.size == 0:
                continue

            roi_frame = local_frame[roi_top:roi_bottom, roi_left:roi_right]
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_frame is None:
                previous_frame = gray_blur
                continue

            frame_delta = cv2.absdiff(previous_frame, gray_blur)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) > 5000 for c in contours)
            now = time.time()
            if motion_detected and (now - last_motion_alert_time > motion_alert_interval):
                last_motion_alert_time = now
                alert_path = os.path.join('motion_alerts', f'motion_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
                cv2.imwrite(alert_path, local_frame)
                with app.app_context():
                    if user:
                        send_email_with_snapshot(user.email, alert_path, detection_type="motion")

            previous_frame = gray_blur
            time.sleep(1)
        except Exception as e:
            print("💥 Motion detection error:", e)
            traceback.print_exc()

# ---------------- Face Detection ---------------- #
def recognize_faces(rgb_frame, known_embeddings, known_names):
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return None
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if face_encodings:
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_embeddings, encoding, tolerance=0.6)
            if True in matches:
                return known_names[matches.index(True)]
    return None

def face_detection_loop(app, user_email):
    global motion_thread_running, frame_buffer

    with app.app_context():
        user = get_user_by_email(user_email)
        db_faces = load_user_face_db(user.id)
        known_embeddings, known_names = load_known_faces(db_faces)

    if not known_embeddings:
        return

    last_face_alert_time = 0
    face_alert_interval = 30

    while motion_thread_running:
        try:
            with frame_lock:
                local_frame = frame_buffer.copy() if frame_buffer is not None else None
            if local_frame is None or local_frame.size == 0:
                continue

            matched_name = recognize_faces(local_frame, known_embeddings, known_names)
            if matched_name:
                now = time.time()
                if now - last_face_alert_time > face_alert_interval:
                    last_face_alert_time = now
                    face_path = os.path.join('motion_alerts', f"face_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(face_path, local_frame)
                    with app.app_context():
                        send_email_with_snapshot(user.email, face_path, detection_type="face")
        except Exception as e:
            print("💥 Face detection error:", e)
            traceback.print_exc()

# ---------------- Vehicle addition ---------------- #
# ---------------- Vehicle addition ---------------- #
@surveillance.route('/add_plate', methods=['POST'])
@token_required
def add_plate(current_user):
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({'msg': 'Missing plate name or image'}), 400

    file = request.files['image']
    name = request.form['name'].strip().lower()

    if file.filename == '':
        return jsonify({'msg': 'No selected file'}), 400

    if len(name) < 5:
        return jsonify({'msg': 'Plate name too short or unreliable'}), 400

    # Check for duplicates (case-insensitive)
    existing_plate = VehiclePlate.query.filter(
        db.func.lower(VehiclePlate.name) == name
    ).first()

    if existing_plate:
        return jsonify({'msg': 'Plate already exists'}), 409

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print(f"✅ New Plate Added: '{name}' with image {filename}")

    new_vehicle_plate = VehiclePlate(
        name=name,
        image_path=filepath
    )
    db.session.add(new_vehicle_plate)
    db.session.commit()

    return jsonify({'msg': 'Vehicle plate data saved successfully.'}), 201

# ---------------- Vehicle Plate Detection ---------------- 


def vehicle_detection_loop(app, user_email):
    global vehicle_thread_running, frame_buffer

    print("🚗 [Vehicle Detection] Loop started...")

    with app.app_context():
        user = get_user_by_email(user_email)
        db_plates = VehiclePlate.query.all()
        stored_plate_names = [plate.name.lower() for plate in db_plates]

    last_vehicle_alert_time = 0
    vehicle_alert_interval = 30  # seconds

    while vehicle_thread_running:
        try:
            with frame_lock:
                local_frame = frame_buffer.copy() if frame_buffer is not None else None
            if local_frame is None or local_frame.size == 0:
                continue

            # Run OCR on the current frame
            results = ocr_reader.readtext(local_frame)
            if not results:
                continue

            print(f"🔍 [Vehicle Detection] OCR Results Found: {[r[1] for r in results]}")

            # Combine all OCR results into one string
            combined_text = ''.join([r[1] for r in results]).replace(" ", "").lower()
            print(f"📷 [Vehicle Detection] Combined detected text: '{combined_text}'")

            # Now compare with stored plates using fuzzy matching
            for plate in stored_plate_names:
                match_score = fuzz.partial_ratio(plate, combined_text)  # Get match score
                if match_score >= 80:  # You can adjust this threshold based on your needs
                    print(f"✅ [Vehicle Detection] MATCHED plate: '{plate}' with score {match_score}")

                    now = time.time()
                    if now - last_vehicle_alert_time > vehicle_alert_interval:
                        last_vehicle_alert_time = now
                        alert_path = os.path.join('motion_alerts', f"vehicle_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(alert_path, local_frame)

                        print(f"📤 [Vehicle Detection] Sending email alert with snapshot: {alert_path}")
                        with app.app_context():
                            send_email_with_snapshot(user.email, alert_path, detection_type="plate")
                    break  # Exit after the first match

        except Exception as e:
            print("💥 [Vehicle Detection] Error:", e)
            traceback.print_exc()
