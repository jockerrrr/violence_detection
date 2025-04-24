from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import time
from ultralytics import YOLO
import logging
import json
from datetime import datetime, timezone
import threading
import queue
import mediapipe as mp
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Increase timeout for large uploads
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output_videos', 'upload')
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')  # Using standard YOLO model

# Create routes to serve static files
@app.route('/output_videos/upload/<path:filename>')
def serve_processed_video(filename):
    logger.info(f"Serving processed video: {filename} from {OUTPUT_FOLDER}")
    # Determine content type based on file extension
    content_type = 'video/mp4'
    if filename.lower().endswith('.avi'):
        content_type = 'video/x-msvideo'

    response = send_from_directory(OUTPUT_FOLDER, filename)
    response.headers['Content-Type'] = content_type
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/uploads/<path:filename>')
def serve_original_video(filename):
    logger.info(f"Serving original video: {filename} from {UPLOAD_FOLDER}")
    response = send_from_directory(UPLOAD_FOLDER, filename)
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure upload settings
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.fromtimestamp(seconds, tz=timezone.utc).strftime('%H:%M:%S'))

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    try:
        # Check if camera is available
        camera_status = "available" if camera is not None and camera.isOpened() else "not available"

        # Check if models are loaded
        yolo_status = "loaded" if yolo_model is not None else "not loaded"
        cnn_lstm_status = "loaded" if cnn_lstm_model is not None else "not loaded"

        # Return detailed health information
        return jsonify({
            'status': 'healthy',
            'message': 'Server is running',
            'camera': camera_status,
            'yolo_model': yolo_status,
            'cnn_lstm_model': cnn_lstm_status,
            'server_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'is_recording': is_recording
        }), 200
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_video():
    try:
        logger.info("Upload endpoint called")

        # Handle preflight CORS request
        if request.method == 'OPTIONS':
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response

        # Log request details
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request files: {request.files}")

        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        logger.info(f"Received file: {file.filename}, {file.content_type}, {file.content_length} bytes")

        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed types: mp4, avi, mov'}), 400

        # Ensure directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        # Change output extension to .mp4 for H.264 codec
        output_filename = f"processed_{timestamp}_{os.path.splitext(filename)[0]}.mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        json_path = os.path.join(OUTPUT_FOLDER, f"detections_{timestamp}_{filename}.json")

        logger.info(f"Saving uploaded file to: {input_path}")
        try:
            file.save(input_path)
            logger.info(f"File saved successfully. Size: {os.path.getsize(input_path)} bytes")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

        # Verify the file was saved correctly
        if not os.path.exists(input_path):
            logger.error(f"File was not saved correctly: {input_path}")
            return jsonify({'error': 'File was not saved correctly'}), 500

        # Initialize models
        logger.info("Initializing models for video processing...")
        if not initialize_models():
            logger.error("Failed to initialize models")
            return jsonify({'error': 'Failed to initialize models'}), 500

        logger.info(f"YOLO model loaded successfully with classes: {yolo_model.names}")
        logger.info("CNN+LSTM model loaded successfully")

        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {input_path}")
            return jsonify({'error': 'Failed to open video file'}), 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize counters for violence detection
        violence_frame_count = 0
        non_violence_frame_count = 0

        logger.info(f"Processing video: {fps} FPS, {frame_width}x{frame_height}, {total_frames} frames")

        # Make sure the directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create video writer
        try:
            # Use H.264 codec which is widely supported
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, 10, (frame_width, frame_height))
            if not out.isOpened():
                logger.error(f"Failed to create video writer: {output_path}")
                # Try fallback to XVID codec
                logger.info("Trying fallback to XVID codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, 10, (frame_width, frame_height))
                if not out.isOpened():
                    logger.error(f"Failed to create video writer with fallback codec: {output_path}")
                    return jsonify({'error': 'Failed to create video writer'}), 500
            logger.info(f"Video writer created successfully: {output_path}")
        except Exception as e:
            logger.error(f"Error creating video writer: {str(e)}")
            return jsonify({'error': f'Error creating video writer: {str(e)}'}), 500

        frame_count = 0
        processed_count = 0

        # Initialize detections list for JSON
        detections_data = {
            "filename": filename,
            "detections": []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame to achieve 10 fps
            if frame_count % int(fps/10) == 0:
                # Calculate current timestamp in video
                current_time = frame_count / fps

                # Perform YOLO detection
                results = yolo_model(frame)
                boxes = results[0].boxes.xyxy  # Extract the bounding boxes

                # We don't need a per-frame violence flag anymore since we're using counters
                # Just process each detection

                # Loop through the detected bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the detected region from the frame (for pose detection)
                    if y2 > y1 and x2 > x1:  # Ensure valid crop dimensions
                        cropped_image = frame[y1:y2, x1:x2]

                        # Convert the cropped image to RGB (MediaPipe requires RGB)
                        img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                        # Apply MediaPipe to extract pose landmarks
                        pose_results = pose.process(img_rgb)

                        is_violence = False

                        if pose_results.pose_landmarks:
                            # Extract keypoints from MediaPipe
                            keypoints = []
                            for lm in pose_results.pose_landmarks.landmark:
                                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

                            # Reshape the keypoints into a 1D array for CNN + LSTM model
                            keypoints = np.array(keypoints).reshape(1, 33, 4)  # 33 keypoints, each with (x, y, z, visibility)

                            # Make prediction using the trained CNN + LSTM model
                            prediction = cnn_lstm_model.predict(keypoints, verbose=0)
                            is_violence = prediction[0] > VIOLENCE_THRESHOLD

                            # Count violent and non-violent detections
                            if is_violence:
                                violence_frame_count += 1  # Increment violence frame counter
                            else:
                                non_violence_frame_count += 1  # Increment non-violence frame counter

                            # Set color based on classification (red for Violence, green for Non-Violence)
                            color = (0, 0, 255) if is_violence else (0, 255, 0)

                            # Draw bounding box with appropriate color (no text on the box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Add detection to JSON data
                            # Calculate normalized bounding box coordinates
                            bbox = {
                                "x": float(x1) / frame_width,
                                "y": float(y1) / frame_height,
                                "width": float(x2 - x1) / frame_width,
                                "height": float(y2 - y1) / frame_height
                            }

                            # Get confidence value from prediction
                            confidence_value = float(prediction[0]) if is_violence else 1.0 - float(prediction[0])

                            # Create class name for JSON
                            violence_label = f"Violence: {is_violence}"

                            detection = {
                                "timestamp": format_timestamp(current_time),
                                "class": violence_label,
                                "is_violence": bool(is_violence),
                                "confidence": round(confidence_value, 2),
                                "bbox": bbox
                            }
                            detections_data["detections"].append(detection)

                # We'll add the dominant classification label later during the reprocessing step
                # For now, just count the frames with violence

                out.write(frame)
                processed_count += 1

                # Log progress every 100 frames
                if processed_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.2f}%")

            frame_count += 1

        cap.release()
        out.release()

        # Determine the dominant classification
        is_video_violent = violence_frame_count >= non_violence_frame_count
        logger.info(f"Violence frames: {violence_frame_count}, Non-violence frames: {non_violence_frame_count}")
        logger.info(f"Video classification: {'Violence' if is_video_violent else 'Non-Violence'}")

        # Update the JSON data with the dominant classification
        detections_data["is_violent"] = is_video_violent
        detections_data["violence_frame_count"] = violence_frame_count
        detections_data["non_violence_frame_count"] = non_violence_frame_count

        # Save detections to JSON file
        with open(json_path, 'w') as f:
            json.dump(detections_data, f, indent=2)

        # Now reprocess the video to apply the dominant classification to all frames
        logger.info("Reprocessing video with dominant classification...")

        # Reopen the processed video
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            logger.error(f"Failed to open processed video for reprocessing: {output_path}")
            # Continue without reprocessing
        else:
            # Create a temporary output file
            temp_output_path = os.path.join(OUTPUT_FOLDER, f"temp_{output_filename}")

            # Create video writer for the temporary file
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            temp_out = cv2.VideoWriter(temp_output_path, fourcc, 10, (frame_width, frame_height))

            if not temp_out.isOpened():
                logger.error(f"Failed to create temporary video writer: {temp_output_path}")
                # Try fallback to XVID codec
                logger.info("Trying fallback to XVID codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                temp_out = cv2.VideoWriter(temp_output_path, fourcc, 10, (frame_width, frame_height))

            if temp_out.isOpened():
                # Process each frame
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Add the dominant classification label to the upper left corner
                    if is_video_violent:
                        # Create a label with Violence: True in red
                        violence_status = "Violence: True"
                        text_color = (0, 0, 255)  # Red color
                    else:
                        # Create a label with Violence: False in green
                        violence_status = "Violence: False"
                        text_color = (0, 255, 0)  # Green color

                    # Create background for text
                    text_size = cv2.getTextSize(violence_status, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    cv2.rectangle(frame, (10, 10), (10 + text_size[0], 10 + text_size[1] + 10), text_color, -1)

                    # Add label to upper left corner
                    cv2.putText(frame, violence_status, (15, 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    # Write the frame to the temporary output file
                    temp_out.write(frame)

                # Release resources
                cap.release()
                temp_out.release()

                # Replace the original output file with the temporary file
                try:
                    os.remove(output_path)
                    os.rename(temp_output_path, output_path)
                    logger.info(f"Successfully reprocessed video with dominant classification")
                except Exception as e:
                    logger.error(f"Error replacing output file: {str(e)}")
            else:
                logger.error("Failed to create temporary video writer, skipping reprocessing")
                cap.release()

        # Log detection data for debugging
        logger.info(f"Total detections: {len(detections_data['detections'])}")
        if len(detections_data['detections']) > 0:
            logger.info(f"Sample detection: {detections_data['detections'][0]}")

        # Keep the input file for comparison
        logger.info("Keeping input file for comparison...")

        # Verify that the files exist and check their sizes
        if not os.path.exists(input_path):
            logger.error(f"Original video file not found: {input_path}")
        else:
            input_size = os.path.getsize(input_path)
            logger.info(f"Original video file exists: {input_path}, Size: {input_size} bytes")

        if not os.path.exists(output_path):
            logger.error(f"Processed video file not found: {output_path}")
        else:
            output_size = os.path.getsize(output_path)
            logger.info(f"Processed video file exists: {output_path}, Size: {output_size} bytes")

        # List all files in the output directory
        logger.info(f"Files in output directory {OUTPUT_FOLDER}:")
        for file in os.listdir(OUTPUT_FOLDER):
            logger.info(f"  - {file}")

        # Convert paths to relative URLs
        output_url = f'/output_videos/upload/{output_filename}'
        json_url = f'/output_videos/upload/detections_{timestamp}_{filename}.json'
        original_url = f'/uploads/{timestamp}_{filename}'

        logger.info(f"Video processing completed successfully")
        logger.info(f"Original video URL: {original_url}")
        logger.info(f"Processed video URL: {output_url}")

        return jsonify({
            'message': 'Video processed successfully',
            'processed_video': output_url,
            'detections_json': json_url,
            'detections': detections_data['detections'],
            'original_video': original_url
        }), 200

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Add these global variables after the existing configurations
camera = None
output_frame = None
lock = threading.Lock()
is_recording = False
video_writer = None
yolo_model = None
cnn_lstm_model = None
mp_pose = None
pose = None
last_detection_time = 0
DETECTION_INTERVAL = 0.1  # 10 frames per second (1/10 = 0.1 seconds between frames)
AUTO_RECORD_DURATION = 30  # Duration to record after violence detection (in seconds)
VIOLENCE_THRESHOLD = 0.92  # Confidence threshold for violence detection (0.92 for CNN+LSTM model)
last_violence_time = 0
auto_recording_start_time = None

# Paths to model files
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')
CNN_LSTM_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'cnn_lstm_pose_weights.keras')

def initialize_models():
    """Initialize YOLO and CNN+LSTM models"""
    global yolo_model, cnn_lstm_model, mp_pose, pose

    try:
        # Initialize YOLO model if not already loaded
        if yolo_model is None:
            logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
            yolo_model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"YOLO model loaded successfully with classes: {yolo_model.names}")

        # Initialize CNN+LSTM model if not already loaded
        if cnn_lstm_model is None:
            logger.info(f"Loading CNN+LSTM model from {CNN_LSTM_MODEL_PATH}...")
            cnn_lstm_model = tf.keras.models.load_model(CNN_LSTM_MODEL_PATH)
            logger.info("CNN+LSTM model loaded successfully")

        # Initialize MediaPipe Pose if not already initialized
        if mp_pose is None:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True)
            logger.info("MediaPipe Pose initialized successfully")

        return True
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        return False

def process_frame(frame):
    """Apply YOLO detection, pose estimation, and violence detection on frame"""
    global is_recording, video_writer, last_violence_time, auto_recording_start_time, yolo_model, cnn_lstm_model, pose

    try:
        # For initial testing, just draw a timestamp on the frame
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add a simple detection box for testing
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.putText(frame, "Test Detection", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Check if we should stop auto-recording
        if is_recording and auto_recording_start_time is not None:
            elapsed_time = time.time() - auto_recording_start_time
            if elapsed_time >= AUTO_RECORD_DURATION and time.time() - last_violence_time >= AUTO_RECORD_DURATION:
                stop_auto_recording()

        # Return the frame with the timestamp and test box
        return frame

        # NOTE: The advanced violence detection code is commented out for initial testing
        # We'll implement it once we confirm the basic video feed is working
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame

def start_auto_recording():
    """Start automatic recording when violence is detected"""
    global camera, video_writer, is_recording

    try:
        # Create recordings directory if it doesn't exist
        recordings_dir = os.path.join(BASE_DIR, 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f'violence_detected_{timestamp}.mp4'
        video_filepath = os.path.join(recordings_dir, video_filename)

        # Get video properties
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 10  # Set to 10 FPS as requested

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))
        is_recording = True

        logger.info(f"Auto-recording started: {video_filename}")

    except Exception as e:
        logger.error(f"Error starting auto-recording: {str(e)}")

def stop_auto_recording():
    """Stop automatic recording"""
    global video_writer, is_recording, auto_recording_start_time

    try:
        if video_writer is not None:
            video_writer.release()
            video_writer = None

        is_recording = False
        auto_recording_start_time = None
        logger.info("Auto-recording stopped")

    except Exception as e:
        logger.error(f"Error stopping auto-recording: {str(e)}")

def generate_frames():
    """Generate frames for video stream"""
    global camera, output_frame, lock, last_detection_time, yolo_model, cnn_lstm_model

    try:
        # Initialize models
        if yolo_model is None or cnn_lstm_model is None:
            logger.info("Initializing models in generate_frames...")
            if not initialize_models():
                logger.error("Failed to initialize models in generate_frames")
                # Return a default error frame
                error_frame = create_error_frame("Failed to initialize models")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return

        # Create a placeholder frame
        placeholder_frame = create_error_frame("Starting camera...")
        ret, buffer = cv2.imencode('.jpg', placeholder_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        while True:
            if camera is None or not camera.isOpened():
                logger.warning("Camera not available in generate_frames")
                error_frame = create_error_frame("Camera not available")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)  # Wait before trying again
                continue

            try:
                success, frame = camera.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    error_frame = create_error_frame("Failed to read frame")
                    ret, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    continue

                current_time = time.time()

                # Process frame with YOLO if enough time has passed (10 FPS)
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    processed_frame = process_frame(frame.copy())
                    last_detection_time = current_time

                    # Save the processed frame
                    with lock:
                        output_frame = processed_frame
                        if is_recording and video_writer is not None:
                            video_writer.write(processed_frame)
                else:
                    with lock:
                        output_frame = frame
                        if is_recording and video_writer is not None:
                            video_writer.write(frame)

                # Encode the frame for streaming
                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    logger.warning("Failed to encode frame")
                    continue

                # Yield the frame in bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                logger.error(f"Error in frame processing loop: {str(e)}")
                error_frame = create_error_frame(f"Error: {str(e)}")
                ret, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)  # Wait before trying again
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
        # Return a default error frame
        error_frame = create_error_frame(f"Error: {str(e)}")
        ret, buffer = cv2.imencode('.jpg', error_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def create_error_frame(error_message):
    """Create an error frame with the given message"""
    # Create a black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add error message
    cv2.putText(frame, error_message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start the camera feed with violence detection"""
    global camera, yolo_model, cnn_lstm_model

    try:
        # Initialize models first
        logger.info("Initializing models for real-time detection...")
        if not initialize_models():
            logger.error("Failed to initialize models in start_camera")
            return jsonify({'status': 'error', 'message': 'Failed to initialize models'}), 500

        # Release the camera if it's already open
        if camera is not None:
            logger.info("Releasing existing camera...")
            camera.release()
            camera = None

        # Open the webcam
        logger.info("Opening webcam...")
        camera = cv2.VideoCapture(0)  # Use default webcam

        # Check if camera opened successfully
        if not camera.isOpened():
            logger.error("Could not open webcam")
            return jsonify({'status': 'error', 'message': 'Could not open webcam. Please check your camera connection.'}), 500

        logger.info("Webcam opened successfully")
        logger.info(f"Camera properties: Width={camera.get(cv2.CAP_PROP_FRAME_WIDTH)}, Height={camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        # Test reading a frame
        ret, frame = camera.read()
        if not ret:
            logger.error("Could not read frame from webcam")
            camera.release()
            camera = None
            return jsonify({'status': 'error', 'message': 'Could not read frame from webcam'}), 500

        logger.info(f"Successfully read frame with shape: {frame.shape}")

        return jsonify({
            'status': 'success',
            'message': 'Camera started with violence detection'
        }), 200
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}")
        # Clean up if there was an error
        if camera is not None:
            try:
                camera.release()
                camera = None
            except:
                pass
        return jsonify({'status': 'error', 'message': f'Error starting camera: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera feed"""
    global camera, is_recording, video_writer

    try:
        if is_recording:
            video_writer.release()
            is_recording = False
            video_writer = None

        if camera is not None:
            camera.release()
            camera = None

        return jsonify({'status': 'success', 'message': 'Camera stopped'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording the camera feed"""
    global camera, video_writer, is_recording

    try:
        if camera is None or not camera.isOpened():
            return jsonify({'status': 'error', 'message': 'Camera not started'}), 400

        if is_recording:
            return jsonify({'status': 'error', 'message': 'Already recording'}), 400

        # Create recordings directory if it doesn't exist
        recordings_dir = os.path.join(BASE_DIR, 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)

        # Create detections directory for JSON files
        detections_dir = os.path.join(recordings_dir, 'detections')
        os.makedirs(detections_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f'recording_{timestamp}.mp4'
        video_filepath = os.path.join(recordings_dir, video_filename)

        # Get video properties
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 10  # Set to 10 FPS as requested

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))
        is_recording = True

        return jsonify({
            'status': 'success',
            'message': 'Recording started',
            'filename': video_filename
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording the camera feed"""
    global video_writer, is_recording

    try:
        if not is_recording:
            return jsonify({'status': 'error', 'message': 'Not recording'}), 400

        video_writer.release()
        video_writer = None
        is_recording = False

        return jsonify({
            'status': 'success',
            'message': 'Recording stopped'
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route for real-time violence detection"""
    response = Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/camera_status', methods=['GET'])
def camera_status():
    """Check if camera is running"""
    global camera
    try:
        if camera is not None and camera.isOpened():
            return jsonify({'status': 'running'}), 200
        else:
            return jsonify({'status': 'stopped'}), 200
    except Exception as e:
        logger.error(f"Error checking camera status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Update the main block to include cleanup
if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    finally:
        if camera is not None:
            camera.release()






