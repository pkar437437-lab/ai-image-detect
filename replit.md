# AI Image Detective

## Overview

AI Image Detective is a Streamlit-based web application designed to detect whether an image is AI-generated or real. The system uses deep learning models built on ResNet-50 architecture to analyze images and provide confidence scores for predictions. The application includes both single-model and ensemble-model detection capabilities, with persistent storage of detection history through a database backend.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Design Pattern**: Single-page application with cached resources to optimize performance
- **Key Features**:
  - Image upload and preview functionality
  - Real-time detection results with confidence scoring
  - Detection history tracking and visualization
  - Support for both single and ensemble model predictions

### Backend Architecture
- **Deep Learning Framework**: PyTorch for model development and inference
- **Model Architecture**: 
  - Base model: ResNet-50 backbone with custom classification head
  - Binary classification (Real vs AI-generated images)
  - Dropout layers (0.5 and 0.3) for regularization
  - Custom fully connected layers (512 neurons) for feature extraction
- **Ensemble System**: Multiple model aggregation for improved accuracy
- **Caching Strategy**: Streamlit's `@st.cache_resource` decorator for model persistence across sessions

### Data Storage
- **Database**: SQLAlchemy ORM with PostgreSQL backend (via DATABASE_URL environment variable)
- **Schema Design**:
  - `detection_history` table stores:
    - Detection results (prediction, confidence)
    - Image metadata (filename, size, format)
    - Model type (single vs ensemble)
    - Ensemble voting details (when applicable)
    - Timestamps for historical tracking
- **Data Export**: CSV and JSON export capabilities for detection history

### Image Processing Pipeline
- **Preprocessing**:
  - Image resizing to 224x224 (ResNet-50 input size)
  - RGB conversion for consistency
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Batch dimension handling for inference
- **Libraries**: PIL for image handling, torchvision transforms for preprocessing

### AI Detection Approach
- **Problem**: Distinguishing between real and AI-generated images
- **Solution**: Transfer learning with ResNet-50, fine-tuned for binary classification
- **Rationale**: ResNet-50 provides strong feature extraction capabilities, and the custom classification head allows specialization for AI artifact detection
- **Alternatives Considered**: Other CNN architectures (VGG, EfficientNet)
- **Pros**: Proven architecture, good balance of accuracy and inference speed
- **Cons**: Requires significant training data for optimal performance

## External Dependencies

### Core ML/AI Libraries
- **PyTorch**: Deep learning framework for model development and inference
- **torchvision**: Pre-trained models and image transformations
- **NumPy**: Numerical operations and array handling

### Web Framework
- **Streamlit**: Web application framework for interactive UI

### Image Processing
- **PIL (Pillow)**: Image loading and manipulation
- **OpenCV (cv2)**: Advanced image processing capabilities

### Database
- **SQLAlchemy**: ORM for database operations
- **PostgreSQL**: Primary database (configured via DATABASE_URL environment variable)
  - Required environment variable: `DATABASE_URL` must be set for database connectivity

### Model Persistence
- Pre-trained models loaded from local storage or model hub
- Ensemble models require multiple model files for aggregation

### Device Support
- CUDA-enabled GPU support for accelerated inference
- CPU fallback for environments without GPU access