# Liver Fat Grading System

A specialized web-based application for automated liver steatosis (fat) grading using ResNet101. The system analyzes liver histology images and provides a grade from 1-4 based on fat bubble density and overall tissue characteristics.

## Features

- **Specialized Liver Histology Analysis**: Analyzes H&E stained liver images for steatosis grading
- **ResNet101 Integration**: Built with ResNet101 model for precise liver tissue classification
- **Grading System**: 1-4 scale based on overall tissue analysis and fat accumulation patterns
- **Real-time Processing**: Instant results with detailed analysis and confidence scores
- **Tissue Classification**: Classifies liver tissue based on steatosis patterns and tissue characteristics
- **Modern Web Interface**: Clean, responsive design with drag-and-drop file upload

## Classification Targets

### Primary Classification Classes
1. **Grade 1** - Minimal steatosis with normal liver tissue appearance
2. **Grade 2** - Mild steatosis with slight fatty infiltration
3. **Grade 3** - Moderate steatosis with noticeable fatty changes
4. **Grade 4** - Severe steatosis with significant fatty liver disease

### Grading Scale

| Grade | Description | Tissue Characteristics |
|-------|-------------|----------------------|
| 1 | Minimal steatosis | Normal liver appearance, minimal fat accumulation |
| 2 | Mild steatosis | Slight fatty infiltration, preserved tissue structure |
| 3 | Moderate steatosis | Noticeable fatty changes, some tissue disruption |
| 4 | Severe steatosis | Significant fatty liver disease, marked tissue changes |

## Project Structure

```
livergrading/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── train_resnet101.py    # ResNet101 training script
├── check_model_accuracy.py # Model accuracy evaluation
├── resnet101_parameters.py # ResNet101 model parameters
├── liver_classification_resnet101.pth # Trained ResNet101 model
├── resnet101_accuracy_results.txt # Accuracy results
├── list_all_images.py    # Dataset image listing utility
├── dataset_simple/       # Training and validation datasets
│   ├── images/
│   │   ├── train/        # Training images
│   │   └── val/          # Validation images
│   └── labels/
│       ├── train/        # Training labels
│       └── val/          # Validation labels
├── test_images_manual/   # Manual test images
├── uploads/              # Uploaded images (created automatically)
└── templates/
    └── index.html        # Web interface
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- GPU recommended for model training (CUDA compatible)

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd livergrading
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will be ready to use

## Model Training

### Current Models

The project currently includes:
- **ResNet101**: Pre-trained classification model for liver steatosis grading

### Training Your Own Model

1. **Prepare your dataset**:
   - Collect liver histology images (H&E stained)
   - Label images according to steatosis grades (1-4)
   - Organize in appropriate format for ResNet101 classification

2. **Run the training script**:
   ```bash
   python train_resnet101.py
   ```

3. **Check model accuracy**:
   ```bash
   python check_model_accuracy.py
   ```

### Dataset Requirements

- **Image Format**: H&E stained liver histology images
- **Classification Labels**: Grade 1, Grade 2, Grade 3, Grade 4
- **Image Size**: Compatible with ResNet101 input requirements
- **Minimum Dataset**: 100+ labeled images per grade for reliable training

## Model Integration

### Current Status
The application includes:
- ResNet101 classification model for steatosis grading
- Tissue analysis and classification logic

### Model Requirements
- ResNet101 model trained for liver histology classification
- Classes: Grade 1, Grade 2, Grade 3, Grade 4
- Compatible with PyTorch ResNet101 format
- Optimized for H&E stained images

## Usage

1. **Upload a Liver Histology Image**
   - Drag and drop an H&E stained liver image
   - Or click to browse and select a file
   - Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF
   - Maximum file size: 16MB

2. **View Results**
   - The system will analyze the image and display:
     - Grade (1-4) with confidence score
     - Tissue analysis results
     - Classification confidence
     - Overall assessment

3. **Interpret Results**
   - Grade 1-2: Generally normal or mild conditions
   - Grade 3-4: May indicate fatty liver disease requiring medical attention
   - Confidence scores indicate reliability of classification

## Analysis Features

### Tissue Classification
- **Grade Classification**: Automatic assignment of steatosis grade (1-4)
- **Confidence Scoring**: Reliability assessment of classification results
- **Tissue Pattern Analysis**: Analysis of overall tissue characteristics

### Quality Assessment
- Image quality evaluation
- Staining quality assessment
- Focus and tissue integrity checks

## API Endpoints

### Health Check
```
GET /health
```
Returns application status and model loading status.

### File Upload
```
POST /upload
```
Accepts multipart form data with image file.
Returns JSON with classification results including:
- Grade and confidence
- Tissue analysis
- Classification details

## Configuration

### File Upload Settings
- Maximum file size: 16MB
- Allowed extensions: PNG, JPG, JPEG, GIF, BMP, TIFF
- Upload directory: `uploads/` (created automatically)

### Model Settings
- Input image size: Compatible with ResNet101 requirements
- H&E specific preprocessing enabled
- Tissue enhancement for better classification
- Contrast optimization for tissue analysis

### Classification Settings
- Minimum confidence threshold: 0.6
- Grade classification enabled
- Tissue pattern analysis active

## Development

### Adding New Classification Classes
1. Update classification configuration in `app.py`
2. Modify classification logic functions
3. Update training script with new classes

### Customizing Grading Logic
1. Modify classification functions in `app.py`
2. Adjust confidence thresholds
3. Add new grading methods as needed

### Extending Analysis
1. Add new analysis functions in `app.py`
2. Update the web interface to display new metrics
3. Modify tissue analysis algorithms

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check model file paths in the application
   - Ensure model files exist and are accessible
   - Verify model format compatibility

2. **Poor classification accuracy**
   - Check image quality and staining
   - Adjust confidence thresholds
   - Verify preprocessing settings

3. **Training issues**
   - Ensure proper dataset format with correct labels
   - Check label quality and consistency
   - Verify GPU availability for training

### Performance Optimization
- Use GPU acceleration if available
- Optimize image preprocessing for your specific H&E staining
- Consider batch processing for multiple images
- Adjust model input size based on your needs

## Security Considerations

- File upload validation prevents malicious file uploads
- Secure filename handling prevents path traversal
- CORS configuration for web security
- Input sanitization for all user inputs
- Medical data privacy compliance

## Medical Disclaimer

**Important**: This application is designed for research and educational purposes only. Medical diagnosis should always be performed by qualified healthcare professionals. The automated grading system is not a substitute for professional medical evaluation.

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with any applicable licenses for the ResNet101 model and other components.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with liver histology images
5. Submit a pull request

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Ensure all dependencies are properly installed
4. Verify model compatibility with your liver images
5. Test with known H&E stained liver samples

---

**Note**: This application is specifically designed for liver histology analysis and requires H&E stained images for optimal performance. The system performs tissue classification for steatosis grading and does not detect individual cellular structures. Medical diagnosis should always be performed by qualified healthcare professionals.
