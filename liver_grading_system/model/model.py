import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class LiverGradingModel:
    def __init__(self):
        self.image_size = (224, 224)
        self.model = self._build_model()
        
    def _build_model(self):
        """Build and return the ResNet model for fatness regression"""
        # Load ResNet50 model without pre-trained weights
        base_model = ResNet50(
            weights=None,  # No pre-trained weights
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Create the full model
        model = models.Sequential([
            # Preprocessing layer
            layers.Rescaling(1./255),
            
            # Base ResNet model
            base_model,
            
            # Additional layers for our specific task
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            # Output layer for regression (0-100)
            layers.Dense(1, activation='linear')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae']  # Mean Absolute Error
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess the input image"""
        img = Image.open(image_path)
        img = img.resize(self.image_size)
        img_array = np.array(img)
        return img_array
    
    def load_training_data(self, data_dir):
        """Load and preprocess training data"""
        images = []
        scores = []
        
        # Assuming data_dir contains subdirectories named with their fatness scores
        for score_dir in os.listdir(data_dir):
            score_path = os.path.join(data_dir, score_dir)
            if os.path.isdir(score_path):
                try:
                    score = float(score_dir)  # Directory name should be the fatness score
                    for img_name in os.listdir(score_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(score_path, img_name)
                            img_array = self.preprocess_image(img_path)
                            images.append(img_array)
                            scores.append(score)
                except ValueError:
                    print(f"Skipping directory {score_dir} - not a valid score")
        
        return np.array(images), np.array(scores)
    
    def train(self, train_data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model with the provided data"""
        # Load and preprocess training data
        print("Loading training data...")
        X, y = self.load_training_data(train_data_dir)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Create data generators with augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # Create data generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Train the model
        print("Starting model training...")
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )
        
        print("Training completed!")
        return history
    
    def predict(self, image_path):
        """Predict the fatness score (0-100) of the liver image"""
        # Preprocess the image
        processed_image = self.preprocess_image(image_path)
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Get prediction
        prediction = self.model.predict(processed_image)
        # Ensure prediction is between 0 and 100
        fatness_score = np.clip(prediction[0][0], 0, 100)
        
        return float(fatness_score) 