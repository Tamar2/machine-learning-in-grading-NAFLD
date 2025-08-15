import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np

print("🎯 CHECKING RESNET101 MODEL ACCURACY")
print("=" * 45)

def load_model():
    """Load the trained ResNet101 model"""
    try:
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load('liver_classification_resnet101.pth', map_location='cpu'))
        model.eval()
        print("✅ ResNet101 model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def get_validation_images():
    """Get list of validation images"""
    val_dir = "dataset_simple/images/val"
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return []
    
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend([f.name for f in Path(val_dir).glob(ext)])
    
    return sorted(images)

def load_csv_data():
    """Load the CSV file with image grades"""
    # Try different possible locations for the CSV file
    possible_paths = [
        "final_image_grade_match_one_to_one_clean.csv",
        "upload_dataset/excel/final_image_grade_match_one_to_one_clean.csv",
        "upload_dataset/final_image_grade_match_one_to_one_clean.csv"
    ]
    
    csv_file = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_file = path
            break
    
    if csv_file is None:
        print(f"❌ CSV file not found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ CSV loaded from {csv_file}: {len(df)} entries")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image for model"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        print(f"❌ Error preprocessing {image_path}: {e}")
        return None

def predict_grade(model, image_tensor):
    """Predict grade for an image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            grade = predicted_class + 1  # Convert 0-based to 1-based
            return grade, confidence, probabilities[0].tolist()
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return 1, 0.0, [0.25, 0.25, 0.25, 0.25]

def main():
    print("🔍 Starting accuracy check...")
    
    # Load model
    print("\n1. Loading ResNet101 model...")
    model = load_model()
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Load CSV data
    print("\n2. Loading CSV data...")
    df = load_csv_data()
    if df is None:
        print("❌ Failed to load CSV data")
        return
    
    # Get validation images
    print("\n3. Getting validation images...")
    val_images = get_validation_images()
    if not val_images:
        print("❌ No validation images found")
        return
    
    print(f"📊 Found {len(val_images)} validation images")
    
    # Initialize counters
    total_correct = 0
    total_images = 0
    grade_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    grade_total = {1: 0, 2: 0, 3: 0, 4: 0}
    grade_predictions = {1: [], 2: [], 3: [], 4: []}
    
    print(f"\n🔍 Testing model on validation set...")
    print("-" * 50)
    
    # Test each validation image
    for img_name in val_images:
        # Get expected grade from CSV
        match = df[df['image_file'] == img_name]
        if match.empty:
            print(f"⚠️  {img_name}: Grade not found in CSV")
            continue
        
        expected_grade = match.iloc[0]['grade']
        grade_total[expected_grade] += 1
        
        # Load and preprocess image
        image_path = f"dataset_simple/images/val/{img_name}"
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            continue
        
        # Make prediction
        predicted_grade, confidence, probabilities = predict_grade(model, image_tensor)
        
        # Check if correct
        is_correct = predicted_grade == expected_grade
        if is_correct:
            total_correct += 1
            grade_correct[expected_grade] += 1
        
        # Store prediction details
        grade_predictions[expected_grade].append({
            'image': img_name,
            'predicted': predicted_grade,
            'expected': expected_grade,
            'confidence': confidence,
            'correct': is_correct
        })
        
        total_images += 1
        
        # Print progress
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_name}: Expected {expected_grade}, Predicted {predicted_grade} (Confidence: {confidence:.1%})")
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    print(f"\n📊 ACCURACY RESULTS")
    print("=" * 50)
    print(f"🎯 Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_images})")
    
    # Calculate grade-wise accuracy
    print(f"\n📈 Grade-wise Accuracy:")
    print("-" * 30)
    for grade in [1, 2, 3, 4]:
        if grade_total[grade] > 0:
            grade_acc = (grade_correct[grade] / grade_total[grade] * 100)
            print(f"   Grade {grade}: {grade_acc:.2f}% ({grade_correct[grade]}/{grade_total[grade]})")
        else:
            print(f"   Grade {grade}: No images found")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS")
    print("=" * 50)
    
    for grade in [1, 2, 3, 4]:
        if grade_predictions[grade]:
            print(f"\n📊 Grade {grade} Predictions:")
            print("-" * 30)
            
            # Count predictions
            pred_counts = {}
            for pred in grade_predictions[grade]:
                pred_grade = pred['predicted']
                pred_counts[pred_grade] = pred_counts.get(pred_grade, 0) + 1
            
            for pred_grade in [1, 2, 3, 4]:
                count = pred_counts.get(pred_grade, 0)
                percentage = (count / len(grade_predictions[grade]) * 100) if grade_predictions[grade] else 0
                print(f"   Predicted Grade {pred_grade}: {count} images ({percentage:.1f}%)")
            
            # Show incorrect predictions
            incorrect = [p for p in grade_predictions[grade] if not p['correct']]
            if incorrect:
                print(f"\n   ❌ Incorrect predictions ({len(incorrect)}):")
                for pred in incorrect[:5]:  # Show first 5
                    print(f"      {pred['image']}: Expected {pred['expected']}, Got {pred['predicted']} (Conf: {pred['confidence']:.1%})")
                if len(incorrect) > 5:
                    print(f"      ... and {len(incorrect) - 5} more")
    
    # Save results to file
    results_file = "resnet101_accuracy_results.txt"
    with open(results_file, "w") as f:
        f.write("RESNET101 MODEL ACCURACY RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_images})\n\n")
        
        f.write("Grade-wise Accuracy:\n")
        for grade in [1, 2, 3, 4]:
            if grade_total[grade] > 0:
                grade_acc = (grade_correct[grade] / grade_total[grade] * 100)
                f.write(f"Grade {grade}: {grade_acc:.2f}% ({grade_correct[grade]}/{grade_total[grade]})\n")
        
        f.write(f"\nDetailed Results:\n")
        for grade in [1, 2, 3, 4]:
            if grade_predictions[grade]:
                f.write(f"\nGrade {grade}:\n")
                for pred in grade_predictions[grade]:
                    status = "CORRECT" if pred['correct'] else "WRONG"
                    f.write(f"  {pred['image']}: Expected {pred['expected']}, Predicted {pred['predicted']}, Confidence {pred['confidence']:.1%} ({status})\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n✅ Accuracy check completed!")

if __name__ == "__main__":
    main() 