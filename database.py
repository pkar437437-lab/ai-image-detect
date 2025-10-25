import os
import json
import csv
from io import StringIO
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """Manage local JSON file operations for detection history."""
    
    def __init__(self):
        # Create a data directory for storing history
        self.data_dir = Path("detection_data")
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / "detection_history.json"
        
        # Initialize history file if it doesn't exist
        if not self.history_file.exists():
            self._save_history([])
    
    def _load_history(self):
        """Load detection history from JSON file."""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    
    def _save_history(self, history):
        """Save detection history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise e
    
    def add_detection(self, filename, prediction, confidence, is_ensemble=False, 
                     ensemble_details=None, image_size=None, file_format=None):
        """Add a new detection result to the history."""
        try:
            history = self._load_history()
            
            # Generate new ID
            new_id = max([item.get('id', 0) for item in history], default=0) + 1
            
            # Convert numpy types to Python native types for JSON serialization
            prediction = int(prediction)
            confidence = float(confidence)
            
            detection = {
                'id': new_id,
                'filename': str(filename),
                'prediction': prediction,
                'confidence': confidence,
                'is_ensemble': bool(is_ensemble),
                'ensemble_details': ensemble_details,
                'timestamp': datetime.utcnow().isoformat(),
                'image_size': str(image_size) if image_size else None,
                'file_format': str(file_format) if file_format else None
            }
            
            history.append(detection)
            self._save_history(history)
            return new_id
        except Exception as e:
            raise e
    
    def get_all_detections(self, limit=100):
        """Get all detection records, most recent first."""
        history = self._load_history()
        # Sort by timestamp descending and apply limit
        sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Convert to object-like structure for compatibility
        class Detection:
            def __init__(self, data):
                self.id = data.get('id')
                self.filename = data.get('filename')
                self.prediction = data.get('prediction')
                self.confidence = data.get('confidence')
                self.is_ensemble = data.get('is_ensemble', False)
                self.ensemble_details = json.dumps(data.get('ensemble_details')) if data.get('ensemble_details') else None
                self.timestamp = datetime.fromisoformat(data.get('timestamp'))
                self.image_size = data.get('image_size')
                self.file_format = data.get('file_format')
        
        return [Detection(item) for item in sorted_history[:limit]]
    
    def get_detection_by_id(self, detection_id):
        """Get a specific detection by ID."""
        history = self._load_history()
        for item in history:
            if item.get('id') == detection_id:
                class Detection:
                    def __init__(self, data):
                        self.id = data.get('id')
                        self.filename = data.get('filename')
                        self.prediction = data.get('prediction')
                        self.confidence = data.get('confidence')
                        self.is_ensemble = data.get('is_ensemble', False)
                        self.ensemble_details = json.dumps(data.get('ensemble_details')) if data.get('ensemble_details') else None
                        self.timestamp = datetime.fromisoformat(data.get('timestamp'))
                        self.image_size = data.get('image_size')
                        self.file_format = data.get('file_format')
                return Detection(item)
        return None
    
    def get_statistics(self):
        """Get statistics about detections."""
        history = self._load_history()
        
        total = len(history)
        ai_generated = sum(1 for item in history if item.get('prediction') == 1)
        real = sum(1 for item in history if item.get('prediction') == 0)
        ensemble_count = sum(1 for item in history if item.get('is_ensemble', False))
        
        return {
            'total': total,
            'ai_generated': ai_generated,
            'real': real,
            'ensemble_count': ensemble_count
        }
    
    def delete_detection(self, detection_id):
        """Delete a detection record."""
        try:
            history = self._load_history()
            filtered_history = [item for item in history if item.get('id') != detection_id]
            
            if len(filtered_history) < len(history):
                self._save_history(filtered_history)
                return True
            return False
        except Exception as e:
            raise e
    
    def clear_all_history(self):
        """Clear all detection history."""
        try:
            self._save_history([])
            return True
        except Exception as e:
            raise e
    
    def export_to_csv(self):
        """Export all detection history to CSV format."""
        detections = self.get_all_detections(limit=10000)
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'ID', 'Filename', 'Prediction', 'Confidence (%)', 
            'Method', 'Image Size', 'Format', 'Timestamp'
        ])
        
        # Write data
        for det in detections:
            writer.writerow([
                det.id,
                det.filename,
                'AI-Generated' if det.prediction == 1 else 'Real',
                f"{det.confidence * 100:.2f}",
                'Ensemble' if det.is_ensemble else 'Single Model',
                det.image_size or 'N/A',
                det.file_format or 'N/A',
                det.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return output.getvalue()
    
    def export_to_json(self):
        """Export all detection history to JSON format."""
        detections = self.get_all_detections(limit=10000)
        
        data = []
        for det in detections:
            entry = {
                'id': det.id,
                'filename': det.filename,
                'prediction': 'AI-Generated' if det.prediction == 1 else 'Real',
                'confidence': det.confidence * 100,
                'method': 'Ensemble' if det.is_ensemble else 'Single Model',
                'image_size': det.image_size,
                'format': det.file_format,
                'timestamp': det.timestamp.isoformat()
            }
            
            if det.is_ensemble and det.ensemble_details:
                try:
                    entry['ensemble_details'] = json.loads(det.ensemble_details)
                except:
                    pass
            
            data.append(entry)
        
        return json.dumps(data, indent=2)
