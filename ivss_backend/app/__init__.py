from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from flask_migrate import Migrate
from app.config import Config

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)
    CORS(app)
    migrate.init_app(app, db)  # Initialize Flask-Migrate

    # Import blueprints here (not models)
    from app.routes.auth import auth_bp
    from app.routes.surveillance import surveillance as surveillance_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(surveillance_bp)

    # Import models *inside* the app context to register them
    with app.app_context():
        from app.models import models, user  
        db.create_all()  # This can be removed since Flask-Migrate will handle migrations

    return app
