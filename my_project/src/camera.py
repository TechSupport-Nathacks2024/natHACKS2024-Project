from flask import Flask, Response, jsonify
from flask_socketio import SocketIO, emit
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import cv2
import dlib
import time
import math
from threading import Thread
from flask_cors import CORS
from playsound import playsound  # Import playsound module

# Initialize Flask app and SocketIO
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO with CORS support for WebSocket connections
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for SocketIO

LOG_FILE = "alerts_log.txt"  # Log file to store alerts

class VideoCamera:
    def __init__(self):
        try:
            # Initialize video stream
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)  # allow camera to warm up

            # Load Haar cascade and Dlib predictor
            self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
            if self.detector.empty():
                raise ValueError("Haar cascade file not loaded. Check 'haarcascade_frontalface_default.xml'.")

            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

            # Constants
            self.EYE_AR_THRESH = 0.2
            self.EYE_AR_CONSEC_FRAMES = 15
            self.YAWN_THRESH = 20
            self.ALERT_TILT_ANGLE = 30  # degrees

            # States
            self.alarm_status = False
            self.alarm_status2 = False
            self.saying = False
            self.COUNTER = 0
        except Exception as e:
            print(f"Initialization error: {e}")
            self.vs = None

    def __del__(self):
        if self.vs is not None:
            self.vs.stop()

    def alarm(self, msg, sound_file=None):
        # Emit the message to React via WebSocket
        socketio.emit('alert', {'message': msg})
        print(msg)
        self.saying = False

        # Log the alert to the file
        with open(LOG_FILE, "a") as f:
            f.write(f"{msg}\n")

        # Play the sound if a sound file is provided
        if sound_file:
            try:
                playsound(sound_file)
            except Exception as e:
                print(f"Error playing sound: {e}")

        time.sleep(1)
        # Clear the log file after logging the alert
        with open(LOG_FILE, "w") as f:
            f.truncate()

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        return abs(np.mean(top_lip, axis=0)[1] - np.mean(low_lip, axis=0)[1])

    def head_tilt_angle(self, shape):
        chin = shape[8]
        nose = shape[30]
        horizontal_distance = chin[0] - nose[0]
        vertical_distance = chin[1] - nose[1]
        return math.atan2(vertical_distance, horizontal_distance) * 180.0 / math.pi

    def get_frame(self):
        if self.vs is None:
            print("Video stream not initialized.")
            return None

        # Read the frame
        frame = self.vs.read()
        if frame is None:
            print("No frame captured from camera.")
            return None

        frame = cv2.resize(frame, (450, 450))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate metrics
            ear, leftEye, rightEye = self.final_ear(shape)
            lip_distance = self.lip_distance(shape)
            tilt_angle = self.head_tilt_angle(shape)

            # Process drowsiness
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    if not self.alarm_status:
                        self.alarm_status = True
                        t = Thread(target=self.alarm, args=("Drowsiness alert!",))
                        t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER = 0
                self.alarm_status = False

            # Only process yawning if not currently in drowsiness alert
            if not self.alarm_status:
                # Process yawning
                if lip_distance > self.YAWN_THRESH:
                    if not self.alarm_status2:
                        self.alarm_status2 = True
                        # Provide the path to your MP3 file here
                        t = Thread(target=self.alarm, args=(" ", "yawn_alert.wav"))
                        t.start()
                    cv2.putText(frame, " ", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.alarm_status2 = False

            # Process head tilt regardless of drowsiness or yawning
            if abs(tilt_angle) > self.ALERT_TILT_ANGLE:
                cv2.putText(frame, " ", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Frame encoding failed.")
            return None
        return jpeg.tobytes()

@app.route('/get_last_alert', methods=['GET'])
def get_last_alert():
    try:
        with open(LOG_FILE, "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip() if lines else "No alerts yet!"
        return jsonify({'last_alert': last_line}), 200
    except FileNotFoundError:
        return jsonify({'last_alert': "No log file found!"}), 404
    except Exception as e:
        return jsonify({'last_alert': f"Error: {str(e)}"}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    camera = VideoCamera()
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=7000)
