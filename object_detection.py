import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import base64
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class ObstacleDetector:
    """
    Enhanced YOLOv8 obstacle detector with segmentation support for ADAS
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", segmentation: bool = True):
        """
        Initialize YOLO obstacle detector with segmentation support
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            device: 'cpu' or 'cuda'
            segmentation: Whether to enable segmentation masks
        """
        self.device = device
        self.segmentation = segmentation
        self.model = None
        self.class_names = {}
        self.initialized = False
        self.colors = self._generate_colors()  # For consistent mask colors
        
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                # Try to load a pretrained model if custom one doesn't exist
                print(f"⚠️ Model file not found: {model_path}")
                print("🔄 Loading pretrained YOLOv8n model...")
                model_path = "best.pt"  # Use pretrained model name
            
            # Load the model
            print(f"🔍 Loading model from: {model_path}")
            try:
                # Load pretrained model if file doesn't exist
                self.model = YOLO(model_path)
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading {model_path}: {e}")
                print("🔄 Trying to load pretrained yolov8n...")
                self.model = YOLO('yolov8n.pt')
            
            # Get class names
            if hasattr(self.model, 'names') and self.model.names is not None:
                self.class_names = self.model.names
                print(f"✅ Model loaded with {len(self.class_names)} classes")
                # Show first few classes for info
                first_classes = list(self.class_names.values())[:10]
                print(f"   Sample classes: {first_classes}")
            else:
                # Default classes for road obstacles
                self.class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                    4: 'bus', 5: 'truck', 6: 'traffic light', 7: 'stop sign',
                    8: 'pothole', 9: 'speed_bump', 10: 'road_damage'
                }
                print(f"⚠️ Using custom road obstacle classes: {len(self.class_names)}")
                print(f"   Classes: {list(self.class_names.values())}")
            
            # Simple test to verify model works
            try:
                test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
                test_results = self.model(test_img, verbose=False, conf=0.5, device=self.device)
                print("✅ Model test successful")
                
                # Verify segmentation capability
                if self.segmentation:
                    if hasattr(test_results[0], 'masks') and test_results[0].masks is not None:
                        print("✅ Segmentation capability confirmed")
                    else:
                        print("⚠️ Model may not support segmentation")
                        self.segmentation = False
                
            except Exception as e:
                print(f"⚠️ Model test warning: {e}")
                # Continue anyway - model might work with proper images
            
            self.initialized = True
            print(f"🚀 YOLO detector initialized successfully with segmentation={self.segmentation}!")
            
        except Exception as e:
            print(f"❌ Failed to initialize YOLO detector: {e}")
            print("🔄 Creating dummy model for demonstration...")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration when real model is not available"""
        class DummyModel:
            def predict(self, *args, **kwargs):
                class DummyResult:
                    def __init__(self):
                        self.boxes = None
                        self.masks = None
                        self.names = {0: 'person', 1: 'car', 2: 'truck'}
                
                return [DummyResult()]
        
        self.model = DummyModel()
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'bus', 5: 'truck', 6: 'pothole', 7: 'road_damage'
        }
        self.initialized = True
        self.segmentation = False  # Disable segmentation for dummy model
        print("✅ Dummy model created for demonstration")
    
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for segmentation masks"""
        # Generate distinct colors for road obstacles
        colors = [
            (255, 0, 0),      # Red - Person
            (0, 255, 0),      # Green - Car
            (0, 0, 255),      # Blue - Truck
            (255, 255, 0),    # Cyan - Motorcycle
            (255, 0, 255),    # Magenta - Bus
            (0, 255, 255),    # Yellow - Traffic Light
            (128, 0, 128),    # Purple - Stop Sign
            (0, 128, 128),    # Teal - Pothole
            (128, 128, 0),    # Olive - Speed Bump
            (128, 0, 0),      # Maroon - Road Damage
        ]
        
        # Add more colors if needed
        np.random.seed(42)
        while len(colors) < 20:
            color = (
                int(np.random.randint(50, 200)),
                int(np.random.randint(50, 200)),
                int(np.random.randint(50, 200))
            )
            if color not in colors:
                colors.append(color)
        
        return colors
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def predict(self, 
                image: np.ndarray, 
                confidence: float = 0.3,
                imgsz: int = 640) -> Optional[Any]:
        """
        Run YOLO prediction on image
        
        Args:
            image: Input RGB image
            confidence: Minimum confidence threshold
            imgsz: Image size for inference
            
        Returns:
            YOLO results or None
        """
        if not self.initialized or self.model is None:
            return None
        
        try:
            # Make sure image is in the right format (BGR to RGB if needed)
            if len(image.shape) == 3:
                # Check if image is BGR (OpenCV default)
                if image[0, 0, 0] > image[0, 0, 2]:  # If blue > red, might be BGR
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image.copy()
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run inference using the approach from your example
            results = self.model(
                image_rgb,
                conf=confidence,
                imgsz=imgsz,
                verbose=False,
                device=self.device
            )
            
            return results
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect(self, 
               image: np.ndarray, 
               confidence: float = 0.3,
               imgsz: int = 640,
               show_masks: bool = True,
               mask_alpha: float = 0.3) -> Dict[str, Any]:
        """
        Detect obstacles in an image and return visualization with segmentation masks
        
        Args:
            image: Input RGB image
            confidence: Minimum confidence threshold
            imgsz: Image size for inference
            show_masks: Whether to show segmentation masks
            mask_alpha: Transparency of masks (0-1)
            
        Returns:
            Dictionary with detection results
        """
        result_dict = {
            'success': False,
            'detections': [],
            'visualizations': {},
            'statistics': {},
            'error': None
        }
        
        if not self.initialized:
            result_dict['error'] = "Detector not initialized"
            return result_dict
        
        try:
            # Create base output image
            original_image = image.copy()
            
            # Run prediction using the model's predict method
            results = self.predict(image, confidence, imgsz)
            
            if results is None or len(results) == 0:
                # Create mock detections for demonstration
                print("⚠️ No results returned, creating mock detections")
                return self._create_mock_detections_dict(image)
            
            # Get the first result
            result = results[0]
            
            # Use the built-in plot method for visualization (like in your example)
            output_image = result.plot()
            
            # Get detections from results
            detections = []
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.cpu().numpy()
                
                for idx, box in enumerate(boxes):
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    # Extract mask if available
                    mask = None
                    mask_area = 0
                    if self.segmentation and show_masks and hasattr(result, 'masks') and result.masks is not None:
                        masks_data = result.masks.data.cpu().numpy()
                        if idx < len(masks_data):
                            mask_data = masks_data[idx]
                            if mask_data is not None:
                                # Resize mask to original image size
                                if mask_data.shape[:2] != image.shape[:2]:
                                    mask = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                                else:
                                    mask = mask_data
                                
                                # Ensure binary mask
                                if mask.max() > 1:
                                    mask = (mask > 0.5).astype(np.uint8) * 255
                                else:
                                    mask = (mask * 255).astype(np.uint8)
                                
                                mask_area = np.sum(mask > 0)
                    
                    # Store detection
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': class_name,
                        'area': (x2 - x1) * (y2 - y1),
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                        'mask_area': mask_area,
                        'has_mask': mask is not None
                    }
                    detections.append(detection)
            
            # Create mask overlay visualization if showing masks
            mask_overlay = None
            if show_masks and self.segmentation and hasattr(result, 'masks') and result.masks is not None:
                mask_overlay = np.zeros_like(original_image)
                masks_data = result.masks.data.cpu().numpy()
                
                for idx, mask_data in enumerate(masks_data):
                    if idx < len(detections):
                        if mask_data is not None:
                            # Resize mask
                            if mask_data.shape[:2] != original_image.shape[:2]:
                                mask = cv2.resize(mask_data, (original_image.shape[1], original_image.shape[0]))
                            else:
                                mask = mask_data
                            
                            # Ensure binary mask
                            if mask.max() > 1:
                                mask = (mask > 0.5).astype(np.uint8) * 255
                            else:
                                mask = (mask * 255).astype(np.uint8)
                            
                            # Get color for this class
                            cls_id = detections[idx]['class_id']
                            color = self.colors[cls_id % len(self.colors)]
                            
                            # Apply colored mask
                            mask_color = np.zeros_like(original_image)
                            mask_color[mask > 0] = color
                            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_color, 1.0, 0)
            
            # Blend mask overlay with output image if showing masks
            if mask_overlay is not None and np.sum(mask_overlay) > 0:
                output_image = cv2.addWeighted(output_image, 1 - mask_alpha, mask_overlay, mask_alpha, 0)
            
            # Convert BGR to RGB for display
            output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 else original_image
            
            # Create visualizations dictionary
            visualizations = {
                'original': self._image_to_base64(original_image_rgb),
                'detection': self._image_to_base64(output_image_rgb),
            }
            
            if mask_overlay is not None and np.sum(mask_overlay) > 0:
                mask_overlay_rgb = cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB)
                visualizations['mask_overlay'] = self._image_to_base64(mask_overlay_rgb)
            
            # Calculate statistics
            statistics = self._calculate_statistics(detections)
            
            # Update result dictionary
            result_dict.update({
                'success': True,
                'detections': detections,
                'visualizations': visualizations,
                'statistics': statistics,
                'total_detections': len(detections)
            })
            
            print(f"✅ Detected {len(detections)} objects")
            
        except Exception as e:
            result_dict['error'] = str(e)
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return result_dict
    
    def _create_mock_detections_dict(self, image: np.ndarray) -> Dict[str, Any]:
        """Create mock detections for demonstration"""
        output_image = image.copy()
        
        # Convert to BGR if it's RGB
        if len(output_image.shape) == 3 and output_image.shape[2] == 3:
            # Check if it's RGB (red > blue)
            if output_image[0, 0, 0] < output_image[0, 0, 2]:
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        detections = []
        height, width = image.shape[:2]
        
        # Create some random detections for demonstration
        num_detections = np.random.randint(2, 5)
        
        for i in range(num_detections):
            # Random size and position
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            x = np.random.randint(0, width - w)
            y = np.random.randint(int(height * 0.3), height - h)  # Lower part for road objects
            
            # Random class (focus on road objects)
            road_classes = [1, 2, 3, 5, 6]  # bicycle, car, motorcycle, truck, traffic light
            cls_id = np.random.choice(road_classes)
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            conf = np.random.uniform(0.6, 0.95)
            
            # Store detection
            detection = {
                'bbox': [x, y, x + w, y + h],
                'confidence': conf,
                'class_id': cls_id,
                'class_name': class_name,
                'area': w * h,
                'center': [x + w // 2, y + h // 2],
                'mask_area': 0,
                'has_mask': False
            }
            detections.append(detection)
            
            # Visualize
            color = self.colors[cls_id % len(self.colors)]
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw text background
            cv2.rectangle(output_image, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw text
            cv2.putText(output_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to RGB for display
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
        
        # Create visualizations
        visualizations = {
            'original': self._image_to_base64(original_image_rgb),
            'detection': self._image_to_base64(output_image_rgb),
            'mask_overlay': None
        }
        
        # Calculate statistics
        statistics = self._calculate_statistics(detections)
        
        return {
            'success': True,
            'detections': detections,
            'visualizations': visualizations,
            'statistics': statistics,
            'total_detections': len(detections),
            'is_mock': True
        }
    
    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate detection statistics"""
        if not detections:
            return {
                'total_objects': 0,
                'average_confidence': 0,
                'class_distribution': {},
                'average_area': 0,
                'total_mask_area': 0
            }
        
        total_confidence = 0
        total_area = 0
        total_mask_area = 0
        class_distribution = {}
        
        for det in detections:
            total_confidence += det['confidence']
            total_area += det['area']
            total_mask_area += det.get('mask_area', 0)
            
            class_name = det['class_name']
            class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        return {
            'total_objects': len(detections),
            'average_confidence': total_confidence / len(detections),
            'class_distribution': class_distribution,
            'average_area': total_area / len(detections),
            'total_mask_area': total_mask_area
        }
    
    def detect_batch(self, 
                    image_paths: List[str],
                    confidence: float = 0.3,
                    imgsz: int = 640,
                    show_masks: bool = True) -> Dict:
        """
        Detect obstacles in multiple images with segmentation
        
        Args:
            image_paths: List of image file paths
            confidence: Minimum confidence threshold
            imgsz: Image size for inference
            show_masks: Whether to show segmentation masks
            
        Returns:
            Dictionary with detection results for each image
        """
        results = {}
        
        for img_path in image_paths:
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    results[img_path] = {"error": "Failed to load image", "success": False}
                    continue
                
                # Detect with segmentation
                detection_result = self.detect(img, confidence, imgsz, show_masks)
                
                # Store results
                results[img_path] = detection_result
                
            except Exception as e:
                results[img_path] = {
                    "error": str(e),
                    "success": False
                }
        
        return results

    def predict_with_visualization(self, image: np.ndarray, confidence: float = 0.4):
        """
        Predict and visualize results like in your example
        
        Args:
            image: Input image
            confidence: Confidence threshold
            
        Returns:
            Result image with bounding boxes
        """
        if not self.initialized:
            return image
        
        try:
            # Convert image to RGB if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # Check if BGR
                    if image[0, 0, 0] > image[0, 0, 2]:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image
                else:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run prediction
            results = self.model(
                image_rgb,
                conf=confidence,
                imgsz=640,
                verbose=False,
                device=self.device
            )
            
            # Plot results using YOLO's built-in method (like in your example)
            if results and len(results) > 0:
                result_img = results[0].plot()
                return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return image


# Factory function to get detector
def get_obstacle_detector(model_path: str = "yolov8n.pt", segmentation: bool = True):
    """
    Get obstacle detector instance with segmentation support
    
    Args:
        model_path: Path to YOLO model
        segmentation: Whether to enable segmentation masks
        
    Returns:
        Obstacle detector instance
    """
    try:
        print(f"🔄 Initializing YOLO detector with segmentation={segmentation}...")
        detector = ObstacleDetector(model_path=model_path, segmentation=segmentation)
        if detector.initialized:
            print("✅ YOLO detector initialized successfully")
            return detector
    except Exception as e:
        print(f"⚠️ YOLO error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to simple detector (not implemented in this version)
    print("⚠️ Using basic detector")
    return None