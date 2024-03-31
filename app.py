# Initialize Roboflow
# rf = Roboflow(api_key="r2vvbbjYWjnauxI3jpgh")
# project = rf.workspace().project("corrosion-m14nc")
# model = project.version(1).model
import os
from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import cv2
import uuid
import ffmpeg
import subprocess

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="1gJjvuxcNW0dW1jqVOrU")
project = rf.workspace().project("corrosion-hjpx8")
model = project.version("1").model

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'upload')
RESULTS_FOLDER = os.path.join(STATIC_FOLDER, 'result')
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'output')

# Ensure upload, result, and output directories exist
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home.html')
def home():
    return render_template('home.html')

@app.route('/result.html')
def result():
    return render_template('result.html')

@app.route('/predict_images', methods=['POST'])
def predict_images():
    image_file = request.files['image']

    # Generate a unique filename for the uploaded image
    image_filename = str(uuid.uuid4()) + '.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image_file.save(image_path)

    # Perform prediction on the image
    predicted_image = model.predict(image_path)
    predicted_image_filename = f"predicted_{image_filename}"
    predicted_image_path = os.path.join(RESULTS_FOLDER, predicted_image_filename)
    predicted_image.save(predicted_image_path)

    return jsonify({'success': True, 'message': 'Image prediction completed successfully!', 'predicted_image_path': predicted_image_path})

@app.route('/predict_videos', methods=['POST'])
def predict_videos():
    confidence = 40
    video_file = request.files['video']

    # Generate a unique filename for the uploaded video
    video_filename = str(uuid.uuid4()) + '.mp4'
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(video_path)

    # Perform predictions
    frames_directory = os.path.join(RESULTS_FOLDER, 'frames')
    os.makedirs(frames_directory, exist_ok=True)
    num_frames = extract_frames(video_path, frames_directory)
    perform_predictions(frames_directory, RESULTS_FOLDER, confidence)

    # Merge predicted frames into a video
    predicted_frames_directory = os.path.join(RESULTS_FOLDER, 'predicted_frames')
    predicted_video_path = os.path.join(OUTPUT_FOLDER, 'predicted_video.mp4')
    merge_predicted_frames_to_video(predicted_frames_directory, predicted_video_path)


    return jsonify({'success': True, 'message': f'Video prediction completed successfully for {num_frames} frames!', 'predicted_video_path': predicted_video_path})

def extract_frames(video_path, frames_directory):
    os.makedirs(frames_directory, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_filename = os.path.join(frames_directory, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    video_capture.release()
    return frame_count

def perform_predictions(frames_directory, results_directory, confidence):
    os.makedirs(results_directory, exist_ok=True)
    predicted_frames_directory = os.path.join(results_directory, 'predicted_frames')
    os.makedirs(predicted_frames_directory, exist_ok=True)
    
    frame_files = sorted(os.listdir(frames_directory))

    for index, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_directory, frame_file)
        predicted_image_filename = f"prediction_{frame_file}"
        predicted_image_path = os.path.join(predicted_frames_directory, predicted_image_filename)
        predicted_image = model.predict(frame_path, confidence=confidence)
        predicted_image.save(predicted_image_path)

def merge_predicted_frames_to_video(predicted_frames_directory, output_video_path):
    frame_files = sorted(os.listdir(predicted_frames_directory))
    frame = cv2.imread(os.path.join(predicted_frames_directory, frame_files[0]))
    height, width, layers = frame.shape

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(predicted_frames_directory, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

if __name__ == '__main__':
    app.run(debug=True)
