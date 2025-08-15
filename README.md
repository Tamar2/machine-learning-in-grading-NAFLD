

# Liver Fat Grading System

A specialized web-based application for automated liver steatosis (fat) grading using ResNet101. The system analyzes liver histology images and provides a grade from 1-4 based on fat bubble density around central veins.

## Features

- **Specialized Liver Histology Analysis**: Detects central veins and fat bubbles (steatosis) in H&E stained liver images
- **ResNet101 Integration**: Built with ResNet101 model for precise fat bubble detection and classification
- **Grading System**: 1-4 scale based on fat bubble count and spatial distribution
- **Real-time Processing**: Instant results with detailed analysis and confidence scores
- **Spatial Analysis**: Analyzes fat distribution relative to central veins (centrilobular vs periportal)
- **Modern Web Interface**: Clean, responsive design with drag-and-drop file upload



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
   - Annotate central veins and fat bubbles
   - Organize in appropriate format for ResNet101

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
- **Annotation Classes**: central_vein, fat_bubble, hepatocyte, portal_triad
- **Image Size**: Compatible with ResNet101 input requirements
- **Minimum Dataset**: 100+ annotated images for reliable training

## Model Integration

### Current Status
The application includes:
- ResNet101 classification model for steatosis grading
- Specialized detection logic for liver steatosis

### Model Requirements
- ResNet101 model trained for liver histology detection
- Classes: central_vein, fat_bubble, hepatocyte, portal_triad
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
     - Grade (1-4)
     - Fat bubble count and area percentage
     - Spatial distribution analysis
     - Central vein detection
     - Detailed detection information

3. **Interpret Results**
   - Grade 1-2: Generally normal or mild conditions
   - Grade 3-4: May indicate fatty liver disease requiring medical attention
   - Spatial distribution helps identify steatosis patterns

## Analysis Features

### Spatial Distribution Analysis
- **Centrilobular**: Fat bubbles near central veins
- **Periportal**: Fat bubbles near portal areas
- **Midzone**: Fat bubbles in intermediate areas

### Detection Details
- Fat bubble count and individual measurements
- Central vein detection and measurements
- Confidence scores for each detection
- Area calculations and percentages

### Quality Metrics
- Image quality assessment
- Staining quality evaluation
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
Returns JSON with detailed grading results including:
- Grade and confidence
- Fat bubble analysis
- Spatial distribution
- Detection details

## Configuration

### File Upload Settings
- Maximum file size: 16MB
- Allowed extensions: PNG, JPG, JPEG, GIF, BMP, TIFF
- Upload directory: `uploads/` (created automatically)

### Model Settings
- Input image size: Compatible with ResNet101 requirements
- H&E specific preprocessing enabled
- White structure enhancement
- Contrast enhancement for better detection

### Detection Settings
- Central vein minimum confidence: 0.7
- Fat bubble size range: 50-2000 pixels
- Fat bubble minimum confidence: 0.6
- Maximum detections: 50 bubbles per image

## Development

### Adding New Detection Classes
1. Update detection configuration in `app.py`
2. Modify `analyze_detections()` function
3. Update training script with new classes

### Customizing Grading Logic
1. Modify `calculate_steatosis_grade()` in `app.py`
2. Adjust thresholds in grading configuration
3. Add new grading methods as needed

### Extending Analysis
1. Add new analysis functions in `app.py`
2. Update the web interface to display new metrics
3. Modify spatial analysis algorithms

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check model file paths in the application
   - Ensure model files exist and are accessible
   - Verify model format compatibility

2. **Poor detection accuracy**
   - Check image quality and staining
   - Adjust confidence thresholds
   - Verify preprocessing settings

3. **Training issues**
   - Ensure proper dataset format
   - Check annotation quality
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

**Note**: This application is specifically designed for liver histology analysis and requires H&E stained images for optimal performance. Medical diagnosis should always be performed by qualified healthcare professionals.
```



