from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
import os
import cv2
from ultralytics import YOLO
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import subprocess
import threading
from queue import Queue
import time
import json

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Global variables for real-time detection
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)
stop_signal = False

# Load the model
model = YOLO('yolov8s.pt')  # Using the base YOLOv8 model

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def process_image(image_path):
    try:
        # Run prediction
        results = model(image_path)
        predictions = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                prediction = {
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                predictions.append(prediction)
        
        return {'status': 'success', 'predictions': predictions}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def process_frame(frame, draw=True):
    try:
        # Improve image quality
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reduce noise
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=5)  # Enhance contrast

        # Normalize image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run prediction with optimized parameters
        results = model(frame_rgb, 
                       conf=0.4,  # Increased confidence threshold
                       iou=0.45,  # Adjusted IOU threshold
                       agnostic_nms=True,  # Improved NMS
                       max_det=50)  # Increased max detections
        
        predictions = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Filter small detections (likely false positives)
                box_width = x2 - x1
                box_height = y2 - y1
                min_size = 20  # Minimum size threshold
                
                if box_width > min_size and box_height > min_size:
                    prediction = {
                        'class': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': [x1, y1, x2, y2]
                    }
                    predictions.append(prediction)
                
                if draw:
                    # Generate unique color for each class
                    class_id = int(box.cls[0])
                    color = ((class_id * 50) % 255, (class_id * 100) % 255, (class_id * 150) % 255)
                    
                    # Draw bounding box with thicker lines
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Add fancy label background
                    label = f"{prediction['class']} {prediction['confidence']:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, predictions
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return frame, []

def get_available_cameras():
    """Get list of available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append({
                    'id': i,
                    'name': f'Camera {i}'
                })
            cap.release()
    return available_cameras

def capture_frames(camera_id=0):
    global stop_signal
    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Reduced to 720p for stability
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Try to set additional properties, but don't fail if not supported
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except:
            pass
    
    # Initialize frame processing variables
    frame_count = 0
    process_every_n_frames = 1  # Process every frame for better accuracy
    frame_buffer = []  # Buffer for frame averaging
    buffer_size = 2  # Number of frames to average
    
    try:
        while not stop_signal:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Add frame to buffer
            frame_buffer.append(frame)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            if frame_count % process_every_n_frames == 0:
                if not frame_queue.full() and len(frame_buffer) == buffer_size:
                    # Average frames to reduce noise
                    avg_frame = cv2.addWeighted(frame_buffer[0], 0.5, frame_buffer[1], 0.5, 0)
                    
                    # Resize frame to optimal size for YOLO (maintain aspect ratio)
                    target_width = 1024
                    aspect_ratio = avg_frame.shape[1] / avg_frame.shape[0]
                    target_height = int(target_width / aspect_ratio)
                    resized_frame = cv2.resize(avg_frame, (target_width, target_height))
                    
                    # Enhance frame quality
                    enhanced_frame = cv2.convertScaleAbs(resized_frame, alpha=1.2, beta=5)
                    
                    frame_queue.put(enhanced_frame)
    finally:
        cap.release()

def process_frames():
    global stop_signal
    while not stop_signal:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame, predictions = process_frame(frame)
            
            if not result_queue.full():
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                result_queue.put({
                    'frame': frame_bytes,
                    'predictions': predictions
                })

def generate_frames():
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            frame_bytes = result['frame']
            predictions = result['predictions']
            
            # Yield both frame and predictions
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'X-Predictions: ' + json.dumps(predictions).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

def process_video(video_path):
    """Process video with enhanced frame extraction and analysis"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'status': 'error', 'message': 'Could not open video file'}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_predictions = {}
        current_second = 0
        frame_count = 0
        
        # Use temporal smoothing for more stable predictions
        smoothing_window = 3  # Number of frames to consider
        previous_predictions = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process one frame per second with additional frames for validation
            current_time = frame_count / fps
            
            # Process this frame and nearby frames for validation
            if abs(current_time - int(current_time)) < 0.1:  # Process frames near the second mark
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (1024, 576))  # 16:9 aspect ratio
                
                # Process current frame
                _, current_pred = process_frame(frame, draw=False)
                
                # Apply temporal smoothing
                previous_predictions.append(current_pred)
                if len(previous_predictions) > smoothing_window:
                    previous_predictions.pop(0)
                
                # Merge predictions from multiple frames
                merged_predictions = []
                confidence_threshold = 0.45  # Higher threshold for more accurate predictions
                
                # Track objects that appear in multiple frames
                object_count = {}
                
                for pred_list in previous_predictions:
                    for pred in pred_list:
                        obj_key = f"{pred['class']}"
                        if pred['confidence'] > confidence_threshold:
                            if obj_key not in object_count:
                                object_count[obj_key] = {'count': 1, 'total_conf': pred['confidence'], 'bbox': pred['bbox']}
                            else:
                                object_count[obj_key]['count'] += 1
                                object_count[obj_key]['total_conf'] += pred['confidence']
                
                # Only keep predictions that appear in multiple frames
                min_appearances = 2
                for obj_key, data in object_count.items():
                    if data['count'] >= min_appearances:
                        avg_confidence = data['total_conf'] / data['count']
                        if avg_confidence > confidence_threshold:
                            merged_predictions.append({
                                'class': obj_key,
                                'confidence': avg_confidence,
                                'bbox': data['bbox']
                            })
                
                frame_predictions[f"second_{int(current_time)}"] = {
                    'timestamp': int(current_time),
                    'predictions': merged_predictions
                }
                current_second = int(current_time)
            
            frame_count += 1
            
        cap.release()
        return {
            'status': 'success',
            'video_info': {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames
            },
            'predictions': frame_predictions
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if not allowed_image_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process image and get predictions
        result = process_image(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict/video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if not allowed_video_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'})
    
    try:
        # Save video temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process video
        result = process_video(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/available_cameras')
def available_cameras():
    cameras = get_available_cameras()
    return jsonify(cameras)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id=0):
    global stop_signal
    stop_signal = False
    
    # Start capture and processing threads
    capture_thread = threading.Thread(target=capture_frames, args=(camera_id,))
    process_thread = threading.Thread(target=process_frames)
    
    capture_thread.start()
    process_thread.start()
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    global stop_signal
    stop_signal = True
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Create required directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)