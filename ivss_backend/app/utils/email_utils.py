import smtplib
from email.message import EmailMessage
import os

def send_email_with_snapshot(to_email, image_path, detection_type="motion"):
    try:
        EMAIL_ADDRESS = 'naniket508@gmail.com'
        EMAIL_PASSWORD = 'dedhzuestvrfzpsg'

        detection_subjects = {
            "motion": "🚨 Motion Detected!",
            "face": "👤 Known Face Detected!",
            "weapon": "🔫 Weapon Detected!",
            "plate": "🚗 Vehicle Plate Detected!"
        }

        detection_messages = {
            "motion": "Motion has been detected. See the attached snapshot.",
            "face": "A known face has been detected. Snapshot attached.",
            "weapon": "A potential weapon has been detected. See the attached snapshot.",
            "plate": "A vehicle plate has been detected. Snapshot attached."
        }

        subject = detection_subjects.get(detection_type.lower(), "📸 Alert!")
        body = detection_messages.get(detection_type.lower(), "Detection event occurred. Snapshot attached.")
        filename = f"{detection_type.lower()}_snapshot.jpg"

        print(f"Preparing email for {detection_type} to {to_email}")
        print(f"📎 Attaching image: {image_path}")

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg.set_content(body)

        with open(image_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=filename)
            print(f"✅ Image attached")

        print("📡 Connecting to Gmail SMTP...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            print("✅ Logged in to SMTP")
            smtp.send_message(msg)
            print("📨 Email sent")

    except Exception as e:
        print(f"⚠️ Error sending email: {e}")
