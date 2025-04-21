# Liver Fatness Grading System

This project uses deep learning to analyze liver images and provide a fatness score (0-100). It features a user-friendly web interface for medical professionals to upload and analyze liver images.

## Features

- Web interface for easy image upload
- Deep learning model for liver fatness analysis
- Real-time fatness scoring (0-100)
- Professional medical interface
- Secure image processing

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone or download this repository to your computer

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and go to:
```
http://localhost:5000
```

## Project Structure

```
liver_grading_system/
├── app.py                 # Main Flask application
├── model/
│   └── model.py          # Deep learning model implementation
├── templates/
│   └── index.html        # Web interface
├── static/               # Static files (CSS, JS, images)
├── uploads/             # Directory for uploaded images
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Usage

1. Open the web interface in your browser
2. Upload a liver image by:
   - Dragging and dropping the image into the upload area
   - Or clicking the "Select Image" button
3. The system will process the image and return a fatness score (0-100)
4. Results will be displayed on the screen

## Note

This is a demonstration project. The model needs to be trained with appropriate medical data before it can be used for actual liver fatness grading.

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 