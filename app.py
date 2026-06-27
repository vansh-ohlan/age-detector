import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import io
import base64

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model (same architecture used in training) ----
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("age_model_resnet34_v2.pth", map_location=device))
model = model.to(device)
model.eval()

mtcnn = MTCNN(keep_all=False, device=device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict_age(pil_img: Image.Image):
    """Detect a face in pil_img and return (predicted_age, box) or (None, None)."""
    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None:
        return None, None

    box = boxes[0]
    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
    x2 = min(pil_img.width, x2)
    y2 = min(pil_img.height, y2)
    face = pil_img.crop((x1, y1, x2, y2))

    face_tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        age = model(face_tensor).item()

    return age, (x1, y1, x2, y2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    age, box = predict_age(img)

    if age is None:
        return jsonify({"error": "No face detected. Try a clearer, front-facing photo."}), 200

    return jsonify({
        "predicted_age": round(age, 1),
        "box": box
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
