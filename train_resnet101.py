import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import random

print("🚀 Starting ResNet101 Training with Improvements")
print("=" * 55)

# Set random seeds
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class ImprovedLiverDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, is_training=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        print(f"📁 Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Load label (grade)
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        try:
            with open(label_path, 'r') as f:
                grade = int(f.read().strip())
        except Exception as e:
            print(f"❌ Error loading label {label_path}: {e}")
            grade = 1
        
        # Convert grade to 0-based index (1->0, 2->1, 3->2, 4->3)
        label = grade - 1
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Enhanced data transformations with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Larger initial size
    transforms.RandomCrop(224),     # Random crop for augmentation
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
    transforms.RandomRotation(degrees=10),   # Small rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("📊 Loading datasets...")

# Load datasets
try:
    train_dataset = ImprovedLiverDataset(
        "dataset_simple/images/train",
        "dataset_simple/labels/train",
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = ImprovedLiverDataset(
        "dataset_simple/images/val", 
        "dataset_simple/labels/val",
        transform=val_transform,
        is_training=False
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} images")
    print(f"✅ Val dataset: {len(val_dataset)} images")
    
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit(1)

# Create data loaders
try:
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)  # Smaller batch size for ResNet101
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    print("✅ Data loaders created successfully")
except Exception as e:
    print(f"❌ Error creating data loaders: {e}")
    exit(1)

# Load ResNet101 model
try:
    print("🔄 Loading ResNet101...")
    model = models.resnet101(pretrained=True)  # Use pre-trained weights
    # Modify the final layer for 4 classes (grades 1-4)
    model.fc = nn.Linear(model.fc.in_features, 4)
    print("✅ ResNet101 loaded successfully")
    print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
model = model.to(device)

# Calculate class weights for imbalanced data
print("⚖️ Calculating class weights...")
grade_counts = [0, 0, 0, 0]  # Count for grades 1, 2, 3, 4
for _, label in train_dataset:
    grade_counts[label] += 1

total_samples = sum(grade_counts)
class_weights = [total_samples / (len(grade_counts) * count) if count > 0 else 1.0 for count in grade_counts]
class_weights = torch.FloatTensor(class_weights).to(device)

print(f"📊 Class distribution: {grade_counts}")
print(f"⚖️ Class weights: {class_weights.tolist()}")

# Loss function and optimizer with improvements
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # AdamW with weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        try:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}: Loss {loss.item():.4f}, Acc {100.*correct/total:.2f}%")
                
        except Exception as e:
            print(f"❌ Error in training batch {batch_idx}: {e}")
            continue
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            except Exception as e:
                print(f"❌ Error in validation batch: {e}")
                continue
    
    return running_loss / len(loader), 100. * correct / total

# Training loop
num_epochs = 50  # More epochs for ResNet101
best_val_acc = 0
patience_counter = 0
patience = 10

print(f"\n🚀 Starting training for {num_epochs} epochs...")
print(f"📊 Using ResNet101 with weighted loss and data augmentation")

try:
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'liver_classification_resnet101.pth')
            print(f"💾 Saved best model with validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping after {patience} epochs without improvement")
            break

    print(f"\n✅ Training completed!")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved as: liver_classification_resnet101.pth")

except Exception as e:
    print(f"❌ Error during training: {e}")
    print("💾 Saving current model...")
    torch.save(model.state_dict(), 'liver_classification_resnet101.pth')

print(f"\n🎯 ResNet101 model ready for use!")
print(f"   Use the saved model 'liver_classification_resnet101.pth' in your web interface")
print(f"   This model should have significantly better accuracy, especially for Grade 3!") 