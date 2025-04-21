from model.model import LiverGradingModel
import os

def main():
    # Initialize the model
    model = LiverGradingModel()
    
    # Path to your training data
    train_data_dir = "training_data"
    
    # Create training_data directory if it doesn't exist
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
        print(f"Created {train_data_dir} directory. Please organize your images as follows:")
        print("training_data/")
        print("├── 0/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── 25/")
        print("│   ├── image3.jpg")
        print("│   └── image4.jpg")
        print("└── ...")
        return
    
    # Train the model
    print("Starting training process...")
    history = model.train(
        train_data_dir=train_data_dir,
        epochs=50,  # Number of training epochs
        batch_size=32,  # Number of images per batch
        validation_split=0.2  # 20% of data used for validation
    )
    
    print("Training completed! Model saved as 'best_model.h5'")

if __name__ == "__main__":
    main() 