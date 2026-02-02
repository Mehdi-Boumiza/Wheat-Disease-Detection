import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from transformers import ViTForImageClassification
import numpy as np


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


IMAGE_FOLDER = "self_test"

EFFICIENT_MODELS = [
    "best_model_final2.pth",
    "best_model_Thefinale2.pth",
]

VIT_MODEL_PATH = "best_model_vit_stage2.pth"

CLASS_NAMES = [
    'Aphid','Black Rust','Blast','Brown Rust','Common Root Rot',
    'Fusarium Head Blight','Healthy','Leaf Blight','Mildew',
    'Mite','Septoria','Smut','Tan spot','Yellow Rust'
]
NUM_CLASSES = len(CLASS_NAMES)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_efficientnet(path):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

efficient_models = [load_efficientnet(p) for p in EFFICIENT_MODELS]


vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
checkpoint = torch.load(VIT_MODEL_PATH, map_location=device)
vit_model.load_state_dict(checkpoint["model_state_dict"])
vit_model.to(device)
vit_model.eval()

#single image prediction
def predict_all_models(img_path):
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    
    predictions = {}
    
    with torch.no_grad():
        # efficientnets
        for model_name, model in zip(EFFICIENT_MODELS, efficient_models):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = np.argmax(probs)
            predictions[model_name] = (CLASS_NAMES[top_idx], probs[top_idx])
        
        # vit
        vit_logits = vit_model(x).logits
        vit_probs = torch.softmax(vit_logits, dim=1)[0].cpu().numpy()
        top_idx = np.argmax(vit_probs)
        predictions["ViT"] = (CLASS_NAMES[top_idx], vit_probs[top_idx])
    
    return predictions

#run of custom folder and compare
for file in sorted(os.listdir(IMAGE_FOLDER)):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(IMAGE_FOLDER, file)
    preds = predict_all_models(path)
    
    print(f"\n{file}")
    for model_name, (cls, conf) in preds.items():
        print(f"  {model_name}: {cls} ({conf*100:.2f}%)")