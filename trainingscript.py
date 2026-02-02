import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from transformers import ViTForImageClassification 


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AddGaussianNoise:
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, tensor):
        return tensor + self.std * torch.randn_like(tensor)

class ClampTensor:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

# config
data_dir = "/Users/mehdiboumiza/Downloads/wheat project for prod/data to use"  
batch_size = 32
num_epochs_stage1 = 55
num_epochs_stage2 = 50
lr_stage1 = 3e-4
lr_stage2 = 1e-5
warmup_steps = 300
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.7, 1.0),
        ratio=(0.85, 1.15)
    ),
    
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.2,
        hue=0.05
    ),
    
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.15),
    
    transforms.ToTensor(),
    
    transforms.RandomApply([
        AddGaussianNoise(std=0.03)
    ], p=0.15),
    
    ClampTensor(0.0, 1.0),
    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_test_transforms)

num_classes = len(train_dataset.classes)

print(f"classes: {train_dataset.classes}")

class_counts = np.bincount(train_dataset.targets)

max_samples = max(class_counts)
min_samples = min(class_counts)
imbalance_ratio = max_samples / min_samples

print(f"\nimbalance ratio: {imbalance_ratio:.2f}x (max: {max_samples}, min: {min_samples})")

use_sampler = imbalance_ratio > 5.0

if use_sampler:
    print("severe imbalance using weighted random sampler")
    targets = train_dataset.targets
    class_sample_counts = np.bincount(targets)
    median_count = np.median(class_sample_counts)
    weights = np.minimum(median_count / class_sample_counts, 3.0)
    sample_weights = [weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                             num_workers=2, pin_memory=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=False)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       num_workers=2, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=2, pin_memory=False)



model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model(model, criterion, optimizer, num_epochs, phase_name="step", 
                use_warmup=False, warmup_steps=500):
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping = EarlyStopping(patience=9, min_delta=0.002, mode='min')
    
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    total_steps = len(train_loader) * num_epochs
    warmup_scheduler = None
    if use_warmup:
        warmup_scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} ({phase_name})")

        # Training
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).logits  
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if use_warmup and warmup_scheduler and global_step < warmup_steps:
                warmup_scheduler.step()
            
            global_step += 1

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.float() / len(train_dataset)

        print(f"train loss: {epoch_loss:.4f} , train acc: {epoch_acc:.4f}")

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).logits  
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_running_corrects.float() / len(val_dataset)
        
        print(f"Val Loss:   {val_loss:.4f} , Val Acc:   {val_acc:.4f}")

        plateau_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f"best_model_{phase_name}.pth")
            print(f" best model saved (Validation Loss: {val_loss:.4f})")

        if early_stopping(val_loss):
            print(f"\nearly stopping triggered at epoch {epoch+1}")
            print(f"best model was at epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")
            break


if __name__ == "__main__":
    
    for param in model.vit.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    #freeze backbone
    optimizer_stage1 = optim.AdamW(model.classifier.parameters(), lr=lr_stage1, weight_decay=0.01)
    train_model(model, criterion, optimizer_stage1, num_epochs_stage1, 
                phase_name="vit_stage1", use_warmup=True, warmup_steps=warmup_steps)

    
   #unfreeze backbone
    
    checkpoint = torch.load("best_model_vit_stage1.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.AdamW(model.parameters(), lr=lr_stage2, weight_decay=0.01)
    train_model(model, criterion, optimizer_stage2, num_epochs_stage2, 
                phase_name="vit_stage2", use_warmup=False)

    #testing
    
    checkpoint = torch.load("best_model_vit_stage2.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" loaded Stage 2 best model (Val Loss: {checkpoint['val_loss']:.4f})")
    
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits  
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("classification report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=train_dataset.classes, 
                                digits=4))

    print("confusion matrix")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)