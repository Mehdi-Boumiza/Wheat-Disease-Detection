import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig


TARGET_DISEASE = "Septoria"   
MODEL_PATH = "vit_binary_Yellowtings.pth"
IMAGE_FOLDER = "self_test"

CLASS_NAMES = ["Other", TARGET_DISEASE]
NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


vit_config = ViTConfig.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES
)

model = ViTForImageClassification(vit_config)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x).logits
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence


print("predictions:")

for file in sorted(os.listdir(IMAGE_FOLDER)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(IMAGE_FOLDER, file)
    label, conf = predict_image(path)

    print(f"{file:25s} : {label:12s} ({conf*100:.2f}%)")