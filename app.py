import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import time
from datetime import datetime
from model import AIImageDetector, EnsembleAIDetector
from utils import preprocess_image, format_confidence
from database import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="AI Image Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize the model
@st.cache_resource
def load_model():
    """Load and cache the AI detection model."""
    try:
        detector = AIImageDetector()
        detector.load_pretrained_model()
        return detector
    except Exception as e:
        st.error(f"Failed to load detection model: {str(e)}")
        return None

@st.cache_resource
def load_ensemble_model():
    """Load and cache the ensemble model."""
    try:
        ensemble = EnsembleAIDetector()
        ensemble.load_ensemble()
        return ensemble
    except Exception as e:
        st.error(f"Failed to load ensemble model: {str(e)}")
        return None

@st.cache_resource
def get_db_manager():
    """Get database manager instance."""
    try:
        return DatabaseManager()
    except Exception as e:
        st.warning(f"Note: {str(e)}")
        return DatabaseManager()  # Return instance anyway since it uses local storage

def main():
    st.title("🔍 AI Image Detective")
    st.markdown("**Detect AI-generated images with advanced deep learning technology**")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model failed to load. Please refresh the page.")
        return
    
    # Processing mode selection
    mode = st.radio(
        "Select Mode:",
        ["Single Image", "History"],
        horizontal=True,
        help="Choose single image for detailed analysis, batch for multiple images, or history to view past detections"
    )
    
    if mode == "Single Image":
        single_image_mode(model)
    elif mode == "Batch Processing":
        batch_processing_mode(model)
    else:
        history_mode()

def single_image_mode(model):
    selected_models = ['resnet50', 'mobilenet']
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Supported formats: PNG, JPEG, WebP",
            label_visibility="collapsed",
            key="single_uploader"
        )
    
    image = None
    if uploaded_file is not None:
        # Display uploaded image
        try:
            image = Image.open(uploaded_file)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.markdown(f"**Filename:** {uploaded_file.name}")
                st.markdown(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
                st.markdown(f"**Format:** {image.format}")
                st.markdown(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
                
        except Exception as e:
            with col1:
                st.error(f"❌ Error loading image: {str(e)}")
            image = None
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        if uploaded_file is not None and image is not None:
            # Analysis button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Preprocessing
                    status_text.text("🔄 Preprocessing image...")
                    progress_bar.progress(25)
                    time.sleep(0.5)  # Simulate processing time
                    
                    processed_image = preprocess_image(image)
                    
                    # Step 2: Model inference
                    status_text.text("🧠 Running AI detection...")
                    progress_bar.progress(50)
                    time.sleep(1.0)  # Simulate processing time
                    
                    individual_results = None
                    
                    with torch.no_grad():
                            prediction, confidence = model.predict(processed_image)
                    
                    # Step 3: Processing results
                    status_text.text("📊 Processing results...")
                    progress_bar.progress(75)
                    time.sleep(0.5)
                    
                    # Step 4: Generate heatmap visualization (only for single model)
                    status_text.text("🎨 Generating heatmap visualization...")
                    progress_bar.progress(85)
                        
                    heatmap = model.generate_heatmap(processed_image, prediction)
                    overlay_image = model.overlay_heatmap(image, heatmap, alpha=0.5)
                    
                    # Step 5: Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Save to database
                    db = get_db_manager()
                    if db:
                        try:
                            db.add_detection(
                                filename=uploaded_file.name,
                                prediction=prediction,
                                confidence=float(confidence),
                                ensemble_details=None,
                                image_size=f"{image.size[0]}x{image.size[1]}",
                                file_format=image.format
                            )
                        except Exception as e:
                            st.warning(f"Failed to save to history: {str(e)}")
                    
                    # Display results
                    if prediction == 1:  # AI-generated
                        st.error("🤖 **AI-GENERATED IMAGE DETECTED**")
                        st.markdown("This image appears to be artificially generated.")
                    else:  # Real image
                        st.success("📷 **REAL IMAGE DETECTED**")
                        st.markdown("This image appears to be authentic/real.")
                    
                    # Confidence score
                    confidence_percent = confidence * 100
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence_percent:.1f}%",
                        delta=None
                    )
                    
                    # Confidence bar
                    st.progress(float(confidence))
                    
                    # Additional metrics
                    st.subheader("📈 Detailed Analysis")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("AI Probability", f"{confidence_percent:.2f}%")
                    with col4:
                        st.metric("Real Probability", f"{100 - confidence_percent:.2f}%")
                    
                    # Interpretation guide
                    st.subheader("📖 How to Interpret Results")
                    
                    if confidence >= 0.9:
                        st.info("🔴 **Very High Confidence** - The model is very certain about its prediction.")
                    elif confidence >= 0.7:
                        st.info("🟡 **High Confidence** - The model is fairly confident about its prediction.")
                    elif confidence >= 0.5:
                        st.info("🟠 **Moderate Confidence** - The model has some uncertainty in its prediction.")
                    else:
                        st.info("⚪ **Low Confidence** - The model is uncertain. Results should be interpreted carefully.")
                    
                    # Heatmap visualization (only for single model)
                        st.subheader("🎨 Detection Heatmap")
                        st.markdown("Red/yellow regions indicate areas that influenced the AI detection decision")
                        
                        col_heat1, col_heat2 = st.columns(2)
                        with col_heat1:
                            st.markdown("**Heatmap Overlay**")
                            st.image(overlay_image, use_column_width=True)
                        with col_heat2:
                            st.markdown("**Pure Heatmap**")
                            st.image(heatmap, use_column_width=True)
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.markdown(f"**Model Architecture:** ResNet-50 based detection network")
                        st.markdown(f"**Heatmap Method:** GradCAM (Gradient-weighted Class Activation Mapping)")
                        st.markdown(f"**Input Resolution:** 224×224 pixels")
                        st.markdown(f"**Processing Time:** ~'2.5 seconds")
                        st.markdown(f"**Detection Threshold:** 0.5")
                        st.markdown(f"**Raw Confidence Score:** {confidence:.6f}")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Analysis failed: {str(e)}")
        else:
            st.info("👆 Please upload an image to start the analysis")

def batch_processing_mode(model):
    """Handle batch image processing."""
    st.subheader("📤 Upload Multiple Images")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg', 'webp'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis",
        key="batch_uploader"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} image(s) uploaded")
        
        # Analyze button
        if st.button("🚀 Analyze All Images", type="primary", use_container_width=True):
            # Progress tracking
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # Get database manager
            db = get_db_manager()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress_value = (idx) / len(uploaded_files)
                    overall_progress.progress(progress_value)
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Load and process image
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    with torch.no_grad():
                        prediction, confidence = model.predict(processed_image)
                    
                    # Save to database
                    if db:
                        try:
                            db.add_detection(
                                filename=uploaded_file.name,
                                prediction=prediction,
                                confidence=float(confidence),
                                is_ensemble=False,
                                ensemble_details=None,
                                image_size=f"{image.size[0]}x{image.size[1]}",
                                file_format=image.format
                            )
                        except Exception as db_error:
                            pass  # Continue even if database save fails
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'image': image,
                        'prediction': prediction,
                        'confidence': confidence,
                        'success': True
                    })
                    
                except Exception as e:
                    results.append({
                        'filename': uploaded_file.name,
                        'image': None,
                        'prediction': None,
                        'confidence': None,
                        'success': False,
                        'error': str(e)
                    })
            
            # Complete progress
            overall_progress.progress(1.0)
            status_text.text("✅ Batch analysis complete!")
            time.sleep(0.5)
            overall_progress.empty()
            status_text.empty()
            
            # Display summary
            st.subheader("📊 Batch Analysis Summary")
            
            ai_count = sum(1 for r in results if r['success'] and r['prediction'] == 1)
            real_count = sum(1 for r in results if r['success'] and r['prediction'] == 0)
            error_count = sum(1 for r in results if not r['success'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🤖 AI-Generated", ai_count)
            with col2:
                st.metric("📷 Real Images", real_count)
            with col3:
                st.metric("❌ Errors", error_count)
            
            # Display individual results
            st.subheader("📋 Individual Results")
            
            for idx, result in enumerate(results):
                with st.expander(f"{'✅' if result['success'] else '❌'} {result['filename']}", expanded=(idx < 3)):
                    if result['success']:
                        col_img, col_res = st.columns([1, 1])
                        
                        with col_img:
                            st.image(result['image'], use_column_width=True)
                        
                        with col_res:
                            if result['prediction'] == 1:
                                st.error("🤖 **AI-GENERATED**")
                            else:
                                st.success("📷 **REAL IMAGE**")
                            
                            confidence_percent = result['confidence'] * 100
                            st.metric("Confidence", f"{confidence_percent:.1f}%")
                            st.progress(result['confidence'])
                    else:
                        st.error(f"Failed to process: {result['error']}")
    else:
        st.info("👆 Please upload multiple images to start batch analysis")

def history_mode():
    """Display detection history."""
    st.subheader("📜 Detection History")
    
    db = get_db_manager()
    if not db:
        st.error("Database not available")
        return
    
    # Get statistics
    try:
        stats = db.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Detections", stats['total'])
        with col2:
            st.metric("AI-Generated", stats['ai_generated'])
        with col3:
            st.metric("Real Images", stats['real'])
        with col4:
            st.metric("Ensemble Used", stats['ensemble_count'])
        
        st.markdown("---")
        
        # Get all detections
        detections = db.get_all_detections(limit=50)
        
        if not detections:
            st.info("No detection history yet. Start analyzing images to build your history!")
            return
        
        # Display detections in a table-like format
        for detection in detections:
            with st.expander(
                f"{'🤖 AI' if detection.prediction == 1 else '📷 Real'} - {detection.filename} "
                f"({detection.timestamp.strftime('%Y-%m-%d %H:%M')})",
                expanded=False
            ):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.markdown(f"**Filename:** {detection.filename}")
                    st.markdown(f"**Prediction:** {'AI-Generated' if detection.prediction == 1 else 'Real Image'}")
                    st.markdown(f"**Confidence:** {detection.confidence * 100:.1f}%")
                    st.markdown(f"**Method:** {'Ensemble (3 models)' if detection.is_ensemble else 'Single Model'}")
                    if detection.image_size:
                        st.markdown(f"**Image Size:** {detection.image_size}")
                    if detection.file_format:
                        st.markdown(f"**Format:** {detection.file_format}")
                
                with col_b:
                    st.markdown(f"**Date:** {detection.timestamp.strftime('%Y-%m-%d')}")
                    st.markdown(f"**Time:** {detection.timestamp.strftime('%H:%M:%S')}")
                    st.markdown(f"**ID:** #{detection.id}")
                
                # Show ensemble details if available
                if detection.is_ensemble and detection.ensemble_details:
                    import json
                    try:
                        ensemble_data = json.loads(detection.ensemble_details)
                        st.markdown("**Individual Model Results:**")
                        for result in ensemble_data:
                            st.markdown(f"- {result['model']}: {result['prediction']} ({result['confidence']:.1f}%)")
                    except:
                        pass
        
        # Export and clear history section
        st.markdown("---")
        st.subheader("📥 Export & Manage")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            # CSV Export
            csv_data = db.export_to_csv()
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # JSON Export
            json_data = db.export_to_json()
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_exp3:
            # Clear history
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                if db.clear_all_history():
                    st.success("History cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear history")
    
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

if __name__ == "__main__":
    main()
