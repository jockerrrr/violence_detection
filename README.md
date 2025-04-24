# Violence Detection System Setup Guide

## Prerequisites
- Python 3.8+ (https://www.python.org/downloads/)
- Node.js 14+ (https://nodejs.org/)
- Git (https://git-scm.com/downloads)
- CUDA Toolkit 11.0+ (for GPU support) (https://developer.nvidia.com/cuda-toolkit)

## Backend Setup

1. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO weights:
```bash
mkdir weights
# Download your custom weights file to weights/best.pt
```

4. Start backend server:
```bash
python app.py
```

## Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Install development tools:
```bash
npm install -g http-server
```

3. Start frontend server:
```bash
http-server -p 8080
```

## Additional Setup

### CUDA Setup (for GPU support)
1. Install NVIDIA GPU drivers
2. Install CUDA Toolkit 11.0+
3. Install cuDNN

### Environment Variables
Create a `.env` file in the backend directory with the following:
```env
FLASK_APP=app.py
FLASK_ENV=development
DEBUG=True
HOST=0.0.0.0
PORT=5000
MAX_CONTENT_LENGTH=104857600
MODEL_PATH=weights/best.pt
```

### Directory Structure
```
project/
├── backend/
│   ├── venv/
│   ├── weights/
│   │   └── best.pt
│   ├── uploads/
│   ├── output_videos/
│   ├── app.py
│   ├── requirements.txt
│   └── .env
└── frontend/
    ├── css/
    ├── js/
    ├── images/
    └── *.html
```

## Troubleshooting

### Common Issues

1. "No module named 'torch'":
```bash
pip install torch torchvision
```

2. OpenCV issues:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

3. CUDA not found:
```bash
# Check CUDA installation
nvidia-smi
# Install specific torch version with CUDA
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

### Port Issues
If port 5000 is already in use:
1. Change the port in `.env`
2. Update frontend API calls in `upload.js` and `realtime.js`

## Testing the Installation

1. Test backend:
```bash
curl http://localhost:5000/health
```

2. Test frontend:
Open browser and navigate to:
```
http://localhost:8080
```

## Updates and Maintenance

Keep dependencies updated:
```bash
pip freeze > requirements.txt
```

## Security Notes

1. Update Flask-CORS settings for production
2. Set proper file upload limits
3. Implement proper authentication
4. Use HTTPS in production