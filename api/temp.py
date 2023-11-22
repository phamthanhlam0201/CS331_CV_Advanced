from flask import Flask, request
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import time
from flask_cors import CORS
import os
import base64
from flask_socketio import SocketIO, emit
import tensorflow as tf
tf.config.run_functions_eagerly(True)

current_directory = os.path.dirname(os.path.abspath(__file__))
# Define the path to the folder where detected faces will be saved
detected_faces_folder = os.path.join(current_directory, 'image')
os.makedirs(detected_faces_folder, exist_ok=True)

json_file_path = os.path.join(current_directory, 'faces_data.json')
# Kiểm tra xem tệp đã tồn tại hay không
if not os.path.isfile(json_file_path):
    # Nếu không, tạo tệp và ghi nội dung rỗng vào đó
    with open(json_file_path, 'w') as json_file:
        json_file.write('{}')

app = Flask(__name__)
socketio = SocketIO()
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

FEATURE_THRESHOLD = 0.6
CHECK_THRESHOLD = 0.4

def extract_features(img):
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg16_model.predict(img)
    return features.flatten()

def detect_faces(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    return faces

def find_matching_face(features, data):
    for face_id, face_data in data.items():
        existing_features = np.array(face_data['extract_feature'])
        similarity = np.dot(features, existing_features) / (np.linalg.norm(features) * np.linalg.norm(existing_features))
        
        if similarity > FEATURE_THRESHOLD:
            return face_id

    return None

def check(features, data):
    for face_id, face_data in data.items():
        existing_features = np.array(face_data['extract_feature'])
        similarity = np.dot(features, existing_features) / (np.linalg.norm(features) * np.linalg.norm(existing_features))
        
        if similarity > CHECK_THRESHOLD:
            return face_id, similarity

    return None

def save_faces_data(data):
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)

@app.route("/ok")
def main():
    print('APP')
    return 'APP'

@socketio.on('facedetect', namespace='/face-detect')
def detect_face_api(base64Image):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    try:
        img_data = base64.b64decode(base64Image)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)

        faces = detect_faces(img)

        if not faces:
            socketio.emit('no-face',{'error': 'No face detected'}, room=request.sid)

        for face in faces:
            x, y, width, height = face['box']
            detected_face = img[y:y+height, x:x+width]
            features = extract_features(detected_face)

            matching_face_id = find_matching_face(features, data)

            if not matching_face_id:
                face_id = str(len(data) + 1)
                name = "Unknown"  # Replace with the actual name information
                data[face_id] = {
                    'id': face_id,
                    'name': name,
                    'extract_feature': features.tolist(),
                    'timestamp': time.time()
                }

                # Save the detected face image to the folder
                detected_face_filename = f"face_{face_id}.JPEG"
                detected_face_path = os.path.join(detected_faces_folder, detected_face_filename)
                cv2.imwrite(detected_face_path, detected_face)

                # Store the absolute file path in faces_data
            # Convert the image data to base64
                with open(detected_face_path, "rb") as image_file:
                    encoded_image_data = base64.b64encode(image_file.read()).decode('utf-8')

                # Store the base64 encoded image data in faces_data
                data[face_id]['detected_face'] = encoded_image_data

                data[face_id]['status'] = 'Already exists' if matching_face_id else 'Saved successfully'

        save_faces_data(data)
        socketio.emit('Result', {'result':data}, room=request.sid)

    except Exception as e:
        print("Error processing request:", str(e))  # Thêm dòng này để in thông điệp lỗi vào console
        emit('Error', str(e), room=request.sid)
    
@socketio.on('getdata', namespace='/get-data')
def get_detected_faces():
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        socketio.emit('Result', {'result':data}, room=request.sid)
    except Exception as e:
        print("Error processing request:", str(e))  # Thêm dòng này để in thông điệp lỗi vào console
        emit('Error', str(e), room=request.sid)
        
    
@socketio.on('updatenames', namespace='/update-names')
def update_all_face_names(data):
    with open('faces_data.json', 'r') as file:
        face_data = json.load(file)
    try:
        updated_names = data.get('updatedNames', [])

        for update in updated_names:
            face_id = update.get('id')
            new_name = update.get('newName')

            # Update only if the current name is "Unknown"
            if face_data.get(face_id, {}).get('name') == "Unknown":
                # Find the face with the corresponding face_id
                for face_key, face_info in face_data.items():
                    if face_info.get('id') == face_id:
                        face_data[face_key]['name'] = new_name
                        break

        save_faces_data(face_data)
        socketio.emit('Notification',{'message': 'Names updated successfully'})
    except Exception as e:
        print("Error processing request:", str(e))  # Thêm dòng này để in thông điệp lỗi vào console
        emit('Error', str(e), room=request.sid)

@socketio.on('facerecognition', namespace='/face-recognition' )
def recognize_api(base64Image):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    try:
        img_data = base64.b64decode(base64Image)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)

        faces = detect_faces(img)

        if not faces:
            socketio.emit('no-face',{'error': 'No face detected'})

        for face in faces:
            x, y, width, height = face['box']
            detected_face = img[y:y+height, x:x+width]
            features = extract_features(detected_face)

            matching_face_id, simi = check(features, data)

            if matching_face_id:
                id=matching_face_id
                face_info[id] ={'bbox': [x,y,width, height], 'name': data[id]['name'], 'similarity':simi}

        socketio.emit('Bbox', 'bbox', face_info)

    except Exception as e:
        print("Error processing request:", str(e))
        emit('Error', str(e), room=request.sid)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='localhost', port=5000)
