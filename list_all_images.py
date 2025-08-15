import os
from pathlib import Path
import pandas as pd

print("📊 COMPLETE IMAGE INVENTORY")
print("=" * 50)

def get_images_in_directory(directory):
    """Get all image files in a directory"""
    if not os.path.exists(directory):
        return []
    
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend([f.name for f in Path(directory).glob(ext)])
    return sorted(images)

def main():
    # Get images from different locations
    train_images = get_images_in_directory("dataset_simple/images/train")
    val_images = get_images_in_directory("dataset_simple/images/val")
    manual_images = get_images_in_directory("test_images_manual")
    original_images = get_images_in_directory("upload_dataset/images")
    
    print(f"\n🎯 TRAINING IMAGES ({len(train_images)} images):")
    print("-" * 40)
    for i, img in enumerate(train_images, 1):
        print(f"{i:3d}. {img}")
    
    print(f"\n🔍 VALIDATION IMAGES ({len(val_images)} images):")
    print("-" * 40)
    for i, img in enumerate(val_images, 1):
        print(f"{i:3d}. {img}")
    
    print(f"\n🧪 MANUAL TEST IMAGES ({len(manual_images)} images):")
    print("-" * 40)
    for i, img in enumerate(manual_images, 1):
        print(f"{i:3d}. {img}")
    
    print(f"\n📁 ORIGINAL UPLOAD IMAGES ({len(original_images)} images):")
    print("-" * 40)
    for i, img in enumerate(original_images, 1):
        print(f"{i:3d}. {img}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print("=" * 30)
    print(f"🏋️ Training set: {len(train_images)} images")
    print(f"🔍 Validation set: {len(val_images)} images")
    print(f"🧪 Manual test set: {len(manual_images)} images")
    print(f"📁 Original upload: {len(original_images)} images")
    
    # Check for any missing images
    all_used = set(train_images + val_images + manual_images)
    missing = set(original_images) - all_used
    
    if missing:
        print(f"\n❓ MISSING IMAGES (not in any dataset):")
        print("-" * 40)
        for i, img in enumerate(sorted(missing), 1):
            print(f"{i:3d}. {img}")
    else:
        print(f"\n✅ All original images are accounted for!")
    
    # Check for duplicates
    all_images = train_images + val_images + manual_images
    duplicates = [img for img in set(all_images) if all_images.count(img) > 1]
    
    if duplicates:
        print(f"\n⚠️ DUPLICATE IMAGES (appear in multiple sets):")
        print("-" * 40)
        for i, img in enumerate(duplicates, 1):
            print(f"{i:3d}. {img}")
    else:
        print(f"\n✅ No duplicate images found!")
    
    # Save detailed report to file
    with open("image_inventory_report.txt", "w") as f:
        f.write("COMPLETE IMAGE INVENTORY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"TRAINING IMAGES ({len(train_images)}):\n")
        f.write("-" * 30 + "\n")
        for img in train_images:
            f.write(f"{img}\n")
        
        f.write(f"\nVALIDATION IMAGES ({len(val_images)}):\n")
        f.write("-" * 30 + "\n")
        for img in val_images:
            f.write(f"{img}\n")
        
        f.write(f"\nMANUAL TEST IMAGES ({len(manual_images)}):\n")
        f.write("-" * 30 + "\n")
        for img in manual_images:
            f.write(f"{img}\n")
        
        f.write(f"\nORIGINAL UPLOAD IMAGES ({len(original_images)}):\n")
        f.write("-" * 30 + "\n")
        for img in original_images:
            f.write(f"{img}\n")
        
        if missing:
            f.write(f"\nMISSING IMAGES:\n")
            f.write("-" * 20 + "\n")
            for img in sorted(missing):
                f.write(f"{img}\n")
        
        if duplicates:
            f.write(f"\nDUPLICATE IMAGES:\n")
            f.write("-" * 20 + "\n")
            for img in duplicates:
                f.write(f"{img}\n")
    
    print(f"\n💾 Detailed report saved to: image_inventory_report.txt")

if __name__ == "__main__":
    main() 