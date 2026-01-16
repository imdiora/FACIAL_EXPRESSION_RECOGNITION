import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

# Import model from your package (no sys.path hacks)
from networks.DDAM import DDAMNet

# -----------------------
# Paths (AUTO from project root)
# -----------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "one_person.mp4")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "rafdb_epoch22_acc0.9055_bacc0.8423.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output_video.avi")

# -----------------------
# Emotion labels (RAF-DB common order)
# If labels look wrong, tell me and I’ll reorder.
# -----------------------
EMOTIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# -----------------------
# Device
# -----------------------
device = torch.device("cpu")  # change to "cuda" if you installed CUDA torch + have GPU

# -----------------------
# Load model
# -----------------------
model = DDAMNet(pretrained=False, num_head=3, num_class=7).to(device)

checkpoint = torch.load(CKPT_PATH, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)

# Remove 'module.' prefix (DataParallel)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

# strict=False allows slight key mismatches (common across versions)
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# -----------------------
# Preprocess
# NOTE: Use 3-channel normalization (your video frames are RGB)
# MixedFeatureNet is typically 112x112. If your training used 224, change to (224,224).
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# -----------------------
# Face detection
# -----------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------
# Video I/O
# -----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps == 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, float(fps), (frame_width, frame_height))

print("Processing video... Press 'q' to quit preview.")

# -----------------------
# Run
# -----------------------
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # BGR -> RGB -> PIL
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            # preprocess -> tensor
            x_tensor = preprocess(face_pil).unsqueeze(0).to(device)  # (1,3,H,W)

            # Forward pass
            out_pred = model(x_tensor)

            # Model might return logits or (logits, ..., ...)
            logits = out_pred[0] if isinstance(out_pred, (tuple, list)) else out_pred

            pred_idx = int(torch.argmax(logits, dim=1).item())
            probs = torch.softmax(logits, dim=1)[0]
            conf = float(probs[pred_idx].item())

            emotion = EMOTIONS[pred_idx] if pred_idx < len(EMOTIONS) else str(pred_idx)
            label = f"{emotion} ({conf:.2f})"

            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

        # Preview window (optional but useful)
        cv2.imshow("Emotion Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done. Output saved to: {OUTPUT_PATH}")
