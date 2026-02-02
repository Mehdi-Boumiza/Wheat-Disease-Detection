import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
import random
import numpy as np
import os

#config
TARGET_DISEASE = "Septoria"  
DATA_DIR = f"/Users/mehdiboumiza/Downloads/wheat project for prod/binary_models/Septoria_binary"
BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 2  


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR,"valid"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"classes: {train_dataset.classes}")

vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=NUM_CLASSES)
model = ViTForImageClassification(vit_config)
model.to(DEVICE)


class_weights = torch.tensor([1.0, 2.0], device=DEVICE)  #boost target class
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR)

if __name__ == "__main__":

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects.float() / len(train_dataset)
        
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss /= len(val_dataset)
        val_acc = val_corrects.float() / len(val_dataset)
        
        print(f"train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # ave best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes
            }, f"vit_binary_septoria.pth")
            print(f" Saved best model for {TARGET_DISEASE} (Val Acc: {val_acc:.4f})")
