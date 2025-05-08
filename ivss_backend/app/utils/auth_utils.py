from flask_jwt_extended import create_access_token
from datetime import timedelta
from app.models.user import User
import face_recognition
import os
from app.models.models import FaceEncoding


def generate_token(identity):
    return create_access_token(identity=identity, expires_delta=timedelta(minutes=30)) 
def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

def load_user_face_db(user_id):
    face_data = []

    faces = FaceEncoding.query.filter_by(user_id=user_id).all()
    for face in faces:
        if os.path.exists(face.image_path):
            image = face_recognition.load_image_file(face.image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                face_data.append((face.name, face.image_path, encoding))
    
    return face_data