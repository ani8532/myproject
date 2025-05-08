from app import db  # Import db here
import numpy as np  # Assuming numpy is used for encoding as an array
from datetime import datetime

class FaceData(db.Model):
    __tablename__ = 'face_data'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Add user_id field
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def __repr__(self):
        return f'<FaceData {self.name}>'

class VehiclePlate(db.Model):
    __tablename__ = 'vehicle_data'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<VehiclePlate {self.name}>'

class FaceEncoding(db.Model):
    __tablename__ = 'face_encodings'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    encoding = db.Column(db.LargeBinary, nullable=False)  # Store face encoding as binary
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Adding user_id to the table
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Now SQLAlchemy can find the 'user' table
    image_path = db.Column(db.String(255), nullable=True)

    def __repr__(self):
        return f"<FaceEncoding(name={self.name}, created_at={self.created_at})>"

