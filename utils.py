import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io

def preprocess_image(image):
    """
    Preprocess an image for model inference.
    
    Args:
        image: PIL Image object
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),          # Convert to tensor and normalize to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply preprocessing
    image_tensor = preprocess(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def format_confidence(confidence):
    """
    Format confidence score for display.
    
    Args:
        confidence: Float confidence score between 0 and 1
        
    Returns:
        str: Formatted confidence string
    """
    percentage = confidence * 100
    return f"{percentage:.1f}%"

def validate_image_format(uploaded_file):
    """
    Validate if the uploaded file is a supported image format.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if f'.{file_extension}' not in allowed_extensions:
            return False, f"Unsupported file format: .{file_extension}"
        
        # Try to open the image to validate it's actually an image
        image = Image.open(uploaded_file)
        
        # Check image size (reasonable limits)
        width, height = image.size
        if width > 4000 or height > 4000:
            return False, "Image too large (max 4000x4000 pixels)"
        
        if width < 32 or height < 32:
            return False, "Image too small (min 32x32 pixels)"
        
        return True, ""
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def calculate_image_stats(image):
    """
    Calculate basic statistics about an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        dict: Dictionary containing image statistics
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    stats = {
        'width': image.size[0],
        'height': image.size[1],
        'channels': len(img_array.shape),
        'mode': image.mode,
        'format': image.format,
        'mean_brightness': np.mean(img_array),
        'std_brightness': np.std(img_array),
    }
    
    # Calculate color channel statistics if RGB
    if image.mode == 'RGB':
        stats['mean_red'] = np.mean(img_array[:, :, 0])
        stats['mean_green'] = np.mean(img_array[:, :, 1])
        stats['mean_blue'] = np.mean(img_array[:, :, 2])
        stats['color_variance'] = np.var(img_array, axis=(0, 1))
    
    return stats

def get_detection_interpretation(confidence, prediction):
    """
    Get human-readable interpretation of detection results.
    
    Args:
        confidence: Float confidence score between 0 and 1
        prediction: Integer prediction (0 for real, 1 for AI-generated)
        
    Returns:
        dict: Dictionary containing interpretation details
    """
    interpretation = {
        'prediction_text': 'AI-Generated' if prediction == 1 else 'Real',
        'confidence_level': '',
        'reliability': '',
        'recommendation': ''
    }
    
    # Determine confidence level
    if confidence >= 0.9:
        interpretation['confidence_level'] = 'Very High'
        interpretation['reliability'] = 'Highly reliable result'
        interpretation['recommendation'] = 'Strong evidence for the prediction'
    elif confidence >= 0.7:
        interpretation['confidence_level'] = 'High'
        interpretation['reliability'] = 'Reliable result'
        interpretation['recommendation'] = 'Good evidence for the prediction'
    elif confidence >= 0.6:
        interpretation['confidence_level'] = 'Moderate'
        interpretation['reliability'] = 'Moderately reliable'
        interpretation['recommendation'] = 'Consider additional verification'
    else:
        interpretation['confidence_level'] = 'Low'
        interpretation['reliability'] = 'Less reliable result'
        interpretation['recommendation'] = 'Results should be interpreted with caution'
    
    return interpretation

def resize_image_for_display(image, max_width=800, max_height=600):
    """
    Resize image for optimal display while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_width: Maximum display width
        max_height: Maximum display height
        
    Returns:
        PIL Image: Resized image
    """
    # Calculate scaling factor
    width_ratio = max_width / image.size[0]
    height_ratio = max_height / image.size[1]
    scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
    
    if scale_factor < 1.0:
        new_width = int(image.size[0] * scale_factor)
        new_height = int(image.size[1] * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def create_confidence_color(confidence):
    """
    Generate a color based on confidence level for UI visualization.
    
    Args:
        confidence: Float confidence score between 0 and 1
        
    Returns:
        str: CSS color string
    """
    if confidence >= 0.8:
        return "#28a745"  # Green for high confidence
    elif confidence >= 0.6:
        return "#ffc107"  # Yellow for moderate confidence
    else:
        return "#dc3545"  # Red for low confidence
