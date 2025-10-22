from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import threading
import queue
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTION_FOLDER'] = 'predictions'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['PREDICTION_FOLDER']).mkdir(exist_ok=True)

# Global variables
cameras = {}
streams = {}

# Initialize models with error handling
def load_models():
    global models, current_model
    models = {}
    
    # First try to load the custom model
    custom_model_path = 'runs/train/weights/best.pt'
    print(f"Attempting to load custom model from: {custom_model_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Load default model first as backup
        models['default'] = YOLO('yolov8s.pt')
        print("Default model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load default model: {str(e)}")
    
    try:
        # Now try to load custom model
        if os.path.exists(custom_model_path):
            models['custom'] = YOLO(custom_model_path)
            print(f"Successfully loaded custom model from {custom_model_path}")
        else:
            raise FileNotFoundError(f"Custom model not found at {custom_model_path}")
    except Exception as e:
        print(f"Error loading custom model: {str(e)}")
        if 'default' in models:
            print("Falling back to default model temporarily")
            models['custom'] = models['default']
        else:
            raise RuntimeError("No models could be loaded")
        
        if not models:
            raise RuntimeError("No models could be loaded")
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        # Ensure we have at least one working model
        if not models:
            raise RuntimeError("No models could be loaded")

# Load models at startup
load_models()

model_conf = {
    'default': 0.45,
    'custom': 0.15  # Lower threshold for better detection of light objects
}
current_model = 'custom'  # Use custom model by default

# Print model status for debugging
print("Current model configuration:")
print(f"- Custom model confidence threshold: {model_conf['custom']}")
print(f"- Default model confidence threshold: {model_conf['default']}")
frame_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

class CameraStream:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.connect_camera()
        
        self.lock = threading.Lock()
        self.frame_rate = 30
        self.last_frame = None
        self.is_running = False
        self.thread = None
        self.predictions = []
        self.frame_count = 0
        self.error_count = 0
        self.last_reconnect_attempt = 0
        
    def connect_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Try DirectShow first (Windows)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                # Fallback to default
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Set camera properties for better performance and stability
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # HD width
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # HD height
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Verify camera is working by reading a test frame
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but unable to read frames")
            
            print(f"Successfully connected to camera {self.camera_id}")
            print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
                
            return True
        except Exception as e:
            print(f"Error connecting to camera {self.camera_id}: {str(e)}")
            return False
        
    def start(self):
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.error_count = 0
                self.frame_count = 0
                self.last_reconnect_attempt = 0
                self.thread = threading.Thread(target=self._capture_loop)
                self.thread.daemon = True
                self.thread.start()
            
    def stop(self):
        with self.lock:
            self.is_running = False
            
        # Wait for thread outside of lock
        if self.thread and threading.current_thread() != self.thread:
            try:
                self.thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
            except Exception as e:
                print(f"Error stopping thread: {str(e)}")
                
        # Final cleanup
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    print(f"Error releasing camera: {str(e)}")
            self.cap = None
            self.last_frame = None
            self.predictions = []
            self.thread = None
            
    def _capture_loop(self):
        last_frame_time = time.time()
        frame_interval = 1.0 / self.frame_rate
        reconnect_interval = 5.0  # Seconds to wait before attempting reconnection
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if we need to reconnect
                if self.cap is None or not self.cap.isOpened():
                    if current_time - self.last_reconnect_attempt >= reconnect_interval:
                        print(f"Attempting to reconnect camera {self.camera_id}")
                        if self.connect_camera():
                            self.error_count = 0
                        self.last_reconnect_attempt = current_time
                    continue
                
                if current_time - last_frame_time >= frame_interval:
                    # Use a try block for the entire frame processing
                    try:
                        success, frame = self.cap.read()
                        if not success:
                            raise Exception("Failed to read frame")
                            
                        # Process frame with error handling
                        self.frame_count += 1
                        
                        # Adaptive frame skipping based on system load
                        skip_frame = False
                        if self.frame_count % 2 != 0:  # Base skip rate
                            skip_frame = True
                        elif self.error_count > 2:  # Increase skip rate on errors
                            skip_frame = self.frame_count % 3 != 0
                            
                        if skip_frame:
                            continue
                        
                        # Enhanced preprocessing for better detection
                        frame = cv2.resize(frame, (1280, 720))  # Ensure consistent size
                        
                        # Convert to LAB color space
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        
                        # Apply CLAHE to L channel
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        
                        # Merge channels
                        limg = cv2.merge((cl,a,b))
                        
                        # Convert back to BGR
                        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                        
                        # Additional enhancements
                        frame = cv2.GaussianBlur(frame, (3, 3), 0)
                        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=-10)
                        
                        with self.lock:
                            # Process frame with YOLO using current model
                            try:
                                results = models[current_model](frame, conf=model_conf[current_model])
                                annotated_frame = results[0].plot()
                                
                                # Extract predictions with more details
                                self.predictions = []
                                for r in results:
                                    boxes = r.boxes
                                    for box in boxes:
                                        pred = {
                                            'class': r.names[int(box.cls[0])],
                                            'confidence': float(box.conf[0]),
                                            'bbox': box.xyxy[0].tolist()
                                        }
                                        self.predictions.append(pred)
                                        
                                self.last_frame = annotated_frame
                                self.error_count = 0  # Reset error count on successful processing
                            except Exception as e:
                                print(f"Model inference error: {str(e)}")
                                self.error_count += 1
                                if self.error_count > 5:  # If too many errors, try to reconnect
                                    self.connect_camera()
                                continue
                        
                        last_frame_time = current_time
                                
                    except Exception as e:
                        print(f"Frame processing error: {str(e)}")
                        self.error_count += 1
                        if self.error_count > 5:
                            self.connect_camera()  # Try to reconnect if too many errors
                            
            except Exception as e:
                print(f"Capture loop error: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error

    def get_frame(self):
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
        return None

    def get_predictions(self):
        with self.lock:
            return self.predictions.copy()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'status': 'error', 'message': 'Could not read image'}
        
        # Enhanced preprocessing for better detection of light objects
        # Convert to LAB color space for better brightness handling
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl,a,b))
        
        # Convert back to BGR
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Additional enhancements
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Adjust contrast and brightness
        image = cv2.convertScaleAbs(image, alpha=1.3, beta=-10)
        
        # Run prediction with optimized parameters using current model
        results = models[current_model](image, conf=model_conf[current_model], iou=0.45)
        predictions = []
        
        # Save annotated image
        filename = os.path.basename(image_path)
        output_path = os.path.join(app.config['PREDICTION_FOLDER'], f'pred_{filename}')
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                pred = {
                    'class': r.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                }
                predictions.append(pred)
        
        # Save the annotated image
        annotated_frame = results[0].plot()
        cv2.imwrite(output_path, annotated_frame)
        
        return {
            'status': 'success',
            'predictions': predictions,
            'annotated_image': f'predictions/pred_{filename}'
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def detect_cameras():
    available_cameras = []
    max_cameras = 2  # Focus on commonly used camera indices
    
    # Try direct indices first (0 and 1 are most common)
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera name if possible
                    camera_name = f"Camera {i}"
                    try:
                        # Try to get the camera name from DirectShow
                        backend_name = cap.getBackendName()
                        if backend_name:
                            camera_name = f"{backend_name} Camera {i}"
                    except:
                        pass
                    
                    available_cameras.append({
                        'id': i,
                        'name': camera_name
                    })
                cap.release()
        except Exception as e:
            print(f"Error detecting camera {i}: {str(e)}")
            continue

    # If no cameras found, try fallback methods
    if not available_cameras:
        try:
            # Try default camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append({
                        'id': 0,
                        'name': "Default Camera"
                    })
                cap.release()
        except Exception as e:
            print(f"Error detecting default camera: {str(e)}")

    print(f"Detected cameras: {available_cameras}")
    return available_cameras

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/video-analysis')
def video_analysis():
    return render_template('video_analysis.html')

@app.route('/available_cameras')
def get_available_cameras():
    cameras = detect_cameras()
    return jsonify(cameras)

def generate_frames(camera_id):
    try:
        stream = streams.get(camera_id)
        if not stream:
            stream = CameraStream(camera_id)
            streams[camera_id] = stream
            stream.start()
        
        error_count = 0
        max_errors = 5
        
        while True:
            try:
                frame = stream.get_frame()
                if frame is None:
                    error_count += 1
                    if error_count > max_errors:
                        print(f"Too many frame errors for camera {camera_id}, stopping stream")
                        break
                    time.sleep(0.1)
                    continue
                
                # Reset error count on successful frame
                error_count = 0
                
                # Ensure frame is valid
                if frame is None:
                    raise Exception("Invalid frame received")
                
                # Resize frame if too large to prevent memory issues
                if frame.shape[0] > 720 or frame.shape[1] > 1280:
                    frame = cv2.resize(frame, (1280, 720))
                elif frame.shape[0] < 480 or frame.shape[1] < 640:
                    # If frame is too small, resize to standard definition
                    frame = cv2.resize(frame, (640, 480))
                
                # Enhance frame quality
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=5)
                
                # Convert frame to JPEG with error handling
                try:
                    ret, buffer = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 85,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
                    if not ret:
                        continue
                    
                    # Convert to bytes and yield
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error in generate_frames: {str(e)}")
                error_count += 1
                if error_count > max_errors:
                    break
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal error in generate_frames: {str(e)}")
    finally:
        # Clean up if the generator exits
        if camera_id in streams:
            streams[camera_id].stop()
            del streams[camera_id]

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    try:
        # Use threading.Lock for thread safety
        with threading.Lock():
            return Response(generate_frames(camera_id),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        return Response("Error: Stream not available", status=500)

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    camera_id = request.json.get('camera_id')
    if camera_id in streams:
        streams[camera_id].stop()
        del streams[camera_id]
    return jsonify({'status': 'success'})

@app.route('/predictions/<int:camera_id>')
def get_predictions(camera_id):
    stream = streams.get(camera_id)
    if stream:
        predictions = stream.get_predictions()
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
    return jsonify({
        'status': 'error',
        'message': 'Camera stream not found'
    })

@app.route('/predict/video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        })
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Create a unique session ID for this video
        session_id = str(int(time.time()))
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_video_realtime, 
                                args=(filepath, session_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message': 'Video processing started'
        })
        
    return jsonify({
        'status': 'error',
        'message': 'Invalid file type'
    })

@app.route('/stream_stats/<int:camera_id>')
def stream_stats(camera_id):
    stream = streams.get(camera_id)
    if stream and stream.cap:
        stats = {
            'fps': stream.frame_rate,
            'resolution': {
                'width': int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
        }
        return jsonify(stats)
    return jsonify({'error': 'Stream not found'})

@app.route('/predict/image', methods=['POST'])
def predict_image_route():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        })
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
        
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type'
        })
        
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        result = process_image(filepath)
        
        # Clean up upload
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/predictions/<path:filename>')
def serve_prediction(filename):
    return send_from_directory(app.config['PREDICTION_FOLDER'], filename)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model
    model_type = request.json.get('model')
    
    try:
        if model_type not in models:
            return jsonify({
                'status': 'error',
                'message': 'Invalid model type'
            })
            
        # Verify model is loaded and working
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        models[model_type](test_img, conf=model_conf[model_type])
        
        # Switch model if test passed
        current_model = model_type
        print(f"Successfully switched to {model_type} model")
        
        return jsonify({
            'status': 'success',
            'message': f'Switched to {model_type} model',
            'current_model': model_type
        })
    except Exception as e:
        print(f"Error switching to {model_type} model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error switching model: {str(e)}'
        })

@app.route('/current_model')
def get_current_model():
    return jsonify({
        'status': 'success',
        'current_model': current_model
    })

# Store video processing sessions
video_sessions = {}

def process_video_realtime(video_path, session_id):
    """Process video in real-time with frame-by-frame analysis"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_sessions[session_id] = {
            'status': 'processing',
            'current_frame': 0,
            'total_frames': total_frames,
            'fps': fps,
            'predictions': [],
            'latest_frame': None,
            'error': None
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame with YOLO
                results = models[current_model](frame, conf=model_conf[current_model])
                annotated_frame = results[0].plot()
                
                # Extract predictions
                frame_predictions = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        pred = {
                            'class': r.names[int(box.cls[0])],
                            'confidence': float(box.conf[0]),
                            'bbox': box.xyxy[0].tolist()
                        }
                        frame_predictions.append(pred)
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Update session data
                video_sessions[session_id].update({
                    'current_frame': frame_count,
                    'latest_frame': frame_bytes,
                    'latest_predictions': frame_predictions
                })
                
                frame_count += 1
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue
        
        # Clean up
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)
            
        video_sessions[session_id]['status'] = 'completed'
        
    except Exception as e:
        video_sessions[session_id].update({
            'status': 'error',
            'error': str(e)
        })
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/video/status/<session_id>')
def get_video_status(session_id):
    """Get current status of video processing"""
    if session_id not in video_sessions:
        return jsonify({
            'status': 'error',
            'message': 'Session not found'
        })
        
    session = video_sessions[session_id]
    return jsonify({
        'status': session['status'],
        'progress': (session['current_frame'] / session['total_frames'] * 100) 
                   if session['total_frames'] > 0 else 0,
        'current_frame': session['current_frame'],
        'total_frames': session['total_frames'],
        'fps': session['fps'],
        'error': session.get('error')
    })

@app.route('/video/frame/<session_id>')
def get_video_frame(session_id):
    """Get latest processed frame"""
    if session_id not in video_sessions:
        return Response('Session not found', status=404)
        
    session = video_sessions[session_id]
    if session['latest_frame'] is None:
        return Response('No frame available', status=404)
        
    return Response(
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + session['latest_frame'] + b'\r\n',
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video/predictions/<session_id>')
def get_video_predictions(session_id):
    """Get latest predictions for the current frame"""
    if session_id not in video_sessions:
        return jsonify({
            'status': 'error',
            'message': 'Session not found'
        })
        
    session = video_sessions[session_id]
    return jsonify({
        'status': 'success',
        'predictions': session.get('latest_predictions', [])
    })

def cleanup_streams():
    """Cleanup function to properly close all camera streams"""
    for camera_id in list(streams.keys()):
        try:
            streams[camera_id].stop()
            del streams[camera_id]
        except Exception as e:
            print(f"Error cleaning up stream {camera_id}: {str(e)}")

import atexit
atexit.register(cleanup_streams)

if __name__ == '__main__':
    try:
        # Check if models are loaded correctly
        if not models:
            print("Error: No models available. Please check model files.")
            exit(1)
            
        print(f"Available models: {list(models.keys())}")
        print(f"Current model: {current_model}")
        print(f"Model configurations: {model_conf}")
            
        # Use production settings with the development server
        print("Starting server...")
        CORS(app)
        app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        cleanup_streams()
    except Exception as e:
        print(f"Server error: {str(e)}")
        cleanup_streams()
        exit(1)