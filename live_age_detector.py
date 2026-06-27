import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Rebuild the same architecture used in training (ResNet34 - improved model)
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("age_model_resnet34_v2.pth", map_location=device))
model = model.to(device)
model.eval()

# Face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    boxes, _ = mtcnn.detect(pil_img)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)

            face = pil_img.crop((x1, y1, x2, y2))
            if face.size[0] == 0 or face.size[1] == 0:
                continue

            face_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_age = model(face_tensor).item()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {predicted_age:.1f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Age Detection (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
   