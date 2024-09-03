import cv2
import face_recognition
from torchvision import transforms
import torch
from .models import Model3



def extract_faces_from_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    face_frames = []

    while True:
        ret, frame = video_capture.read()  # Read a frame from the video stream

        if not ret:
            break  # Break the loop if no more frames are available

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            for face in face_locations:
                top, right, bottom, left = face
                face_frame = frame[top:bottom, left:right]
                face_frame = cv2.resize(face_frame, (128, 128))  # Resize for model input
                face_frames.append(face_frame)

    video_capture.release()
    return face_frames

def preprocess_face_frames(face_frames, device='cpu'):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocessed_frames = torch.stack([preprocess(frame) for frame in face_frames]).to(device)

    # Add batch dimension and sequence length dimension
    preprocessed_frames = preprocessed_frames.unsqueeze(0)  # Shape: (1, seq_length, c, h, w)

    return preprocessed_frames



def predict(model, preprocessed_input):
    with torch.no_grad():  # Disable gradient computation for inference
        fmap, output = model(preprocessed_input)
    return fmap, output

def postprocess_output(output):
    #classification task
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities

def load_model(model_path, num_classes, device='cpu'):
    model = Model3(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def predict_face_video(model_path, face_frames, num_classes, device='cpu'):
    model = load_model(model_path, num_classes, device)

    preprocessed_input = preprocess_face_frames(face_frames, device)

    fmap, output = predict(model, preprocessed_input)

    predicted_class, probabilities = postprocess_output(output)

    return predicted_class, probabilities


def classify_video(video_path, model_path, num_classes, device='cpu'):
    # Step 1: Extract faces from the video
    face_frames = extract_faces_from_video(video_path)

    # Step 2: Predict using the model
    predicted_class, probabilities = predict_face_video(model_path, face_frames, num_classes, device)
    probabilities = probabilities.tolist()
    return predicted_class, probabilities