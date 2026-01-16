import os
import sys
import torch
import cv2
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import torch.nn as nn

# Add the networks directory to the system path
#sys.path.append(r'C:\Users\HP Envy\Desktop\CV projects\DDAMFN\networks')

# Import your DDAMNet model and necessary components
#from DDAM import DDAMNet, CoordAttHead, CoordAtt
from networks.DDAM import DDAMNet, CoordAttHead, CoordAtt


# Define emotion names mapping
emotion_names = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Initialize the model with the correct parameters
model = DDAMNet(pretrained=True, num_head=3, num_class=7)

# Load the checkpoint
model_path = r"C:\Users\HP Envy\Desktop\CV projects\DDAMFN\checkpoints\rafdb_epoch22_acc0.9055_bacc0.8423.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the model state_dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Remove all leading 'module.' prefixes from keys
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        while name.startswith('module.'):
            name = name[len('module.'):]
        new_state_dict[name] = v
    return new_state_dict

clean_state_dict = remove_module_prefix(state_dict)

# Load the state_dict into the model
missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

# Optionally, print missing and unexpected keys
if missing_keys:
    print('Missing keys:', missing_keys)
if unexpected_keys:
    print('Unexpected keys:', unexpected_keys)

model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

print("Starting real-time emotion recognition...")
print("Press 'q' to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Convert to PIL Image
            face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            # Preprocess the face
            face_tensor = preprocess(face_image).unsqueeze(0)

            # Get prediction
            with torch.no_grad():
                prediction, _, _ = model(face_tensor)

                # Get predicted class and confidence
                predicted_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1)[0, predicted_class].item()

                # Get emotion name
                emotion = emotion_names[predicted_class]

                # Create label with emotion and confidence
                label = f"{emotion}: {confidence:.2f}"

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 255, 0), -1)

            # Add text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Display the frame
        cv2.imshow('Real-time Emotion Recognition', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nEmotion recognition stopped.")