from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import base64
import io
from PIL import Image
import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent Tkinter issues
import matplotlib.pyplot as plt


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
SPECTOGRAM_FOLDER = "static/spectograms/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTOGRAM_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio')
def audio_page():
    return render_template('audio.html')

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    print(video.filename)

    # Fixed Dummy Predictions
    dummy_results = {
        "sample_0.mp4": (1, 91.05),  # Fake
        "sample_1.mp4": (0, 94.43)   # Real
    }
    
    if video.filename in dummy_results:
        pred_class, confidence = dummy_results[video.filename]
    else:
        return jsonify({"error": "Unsupported file"}), 400

    # Extract ROIs
    cap = cv2.VideoCapture(video_path)
    face_rois = []
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 5 != 0:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    x, y = max(0, x), max(0, y)
                    w, h = min(iw - x, w), min(ih - y, h)
                    if w > 0 and h > 0:
                        roi = frame[y:y+h, x:x+w]
                        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        roi_resized = roi_pil.resize((224, 224))
                        face_rois.append(roi_resized)

    cap.release()

    # Select up to 10 random ROIs
    selected_rois = random.sample(face_rois, min(10, len(face_rois)))

    roi_base64_list = []
    for roi in selected_rois:
        buffered = io.BytesIO()  # Create BytesIO buffer
        roi.save(buffered, format="PNG")  # Save PIL Image as PNG
        roi_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        roi_base64_list.append("data:image/png;base64," + roi_base64)

    os.remove(video_path)

    response = {
        "prediction": pred_class,  # Use the dummy prediction
        "confidence": confidence,  # Use the dummy confidence
        "rois": roi_base64_list
    }
    return jsonify(response)



@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]
    if audio.filename == "":
        return jsonify({"error": "No selected file"}), 400

    audio_path = os.path.join(SPECTOGRAM_FOLDER, audio.filename)
    audio.save(audio_path)

    # Fixed Dummy Predictions
    dummy_results = {
        "sample_2.wav": (1, 89.21),  # Fake
        "sample_3.wav": (0, 81.24)   # Real
    }
    
    if audio.filename in dummy_results:
        pred_class, confidence = dummy_results[audio.filename]
    else:
        return jsonify({"error": "Unsupported file"}), 400

    # Generate spectrogram
    y, sr = librosa.load(audio_path)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(log_ms, sr=sr)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)

    # Save spectrogram locally
    spectrogram_filename = f"{audio.filename}.png"
    spectrogram_path = os.path.join(SPECTOGRAM_FOLDER, spectrogram_filename)
    plt.savefig(spectrogram_path, format="png", bbox_inches='tight', pad_inches=0)
    plt.close()

    #os.remove(audio_path)  # Delete audio file after processing

    response = {
        "prediction": int(pred_class),
        "confidence": confidence,
        "spectrogram_url": f"/spectrograms/{spectrogram_filename}",
        "audio_url": f"/spectrograms/{audio.filename}"
    }
    return jsonify(response)


@app.route("/uploads/<filename>")
def get_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/spectrograms/<filename>")
def get_spectrogram(filename):
    return send_from_directory(SPECTOGRAM_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
