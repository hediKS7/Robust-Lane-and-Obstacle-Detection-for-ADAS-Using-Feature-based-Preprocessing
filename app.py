import streamlit as st
import numpy as np
import cv2
from PIL import Image
from processing import process_image
from utils import create_sample_image, reset_processing, apply_custom_css
from object_detection import get_obstacle_detector, ObstacleDetector

# Set page configuration - MUST BE FIRST Streamlit command
st.set_page_config(
    page_title="Lane Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS - AFTER set_page_config
from utils import apply_custom_css
apply_custom_css()

# Initialize obstacle detector in session state
if 'obstacle_detector' not in st.session_state:
    st.session_state.obstacle_detector = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# Initialize session state
if 'use_sample' not in st.session_state:
    st.session_state.use_sample = False
if 'sample_mode' not in st.session_state:
    st.session_state.sample_mode = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = 'straight'
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False

# ========== OBSTACLE DETECTION FUNCTIONS ==========

def init_obstacle_detector():
    """Initialize obstacle detector"""
    if st.session_state.obstacle_detector is None:
        with st.spinner("🔄 Loading obstacle detection model..."):
            try:
                detector = get_obstacle_detector(segmentation=True)
                if detector:
                    st.session_state.obstacle_detector = detector
                    st.success("✅ Obstacle detector loaded successfully!")
                else:
                    st.warning("⚠️ Obstacle detector could not be loaded, but lane detection will still work.")
            except Exception as e:
                st.error(f"❌ Error loading obstacle detector: {e}")

def reset_obstacle_detection():
    """Reset obstacle detection state"""
    st.session_state.detection_results = None

def create_obstacle_report(results, image_source):
    """Create obstacle detection report text for download"""
    report_text = f"Obstacle Detection Results Report\n"
    report_text += "=" * 50 + "\n"
    report_text += f"Detection Mode: Obstacle Detection\n"
    report_text += f"Image Source: {image_source}\n"
    
    if results.get('is_mock', False):
        report_text += f"Status: MOCK DETECTIONS (Real model not available)\n\n"
    elif results.get('success', False):
        report_text += f"Status: SUCCESS\n\n"
    else:
        report_text += f"Status: FAILED\n"
        report_text += f"Error: {results.get('error', 'Unknown')}\n\n"
        return report_text
    
    # Add statistics
    statistics = results.get('statistics', {})
    report_text += f"Statistics:\n"
    report_text += f"  - Total Objects: {statistics.get('total_objects', 0)}\n"
    report_text += f"  - Average Confidence: {statistics.get('average_confidence', 0):.2f}\n"
    report_text += f"  - Average Area: {statistics.get('average_area', 0):,.0f} px²\n"
    report_text += f"  - Total Mask Area: {statistics.get('total_mask_area', 0):,.0f} px\n\n"
    
    # Add class distribution
    class_dist = statistics.get('class_distribution', {})
    if class_dist:
        report_text += f"Class Distribution:\n"
        for class_name, count in class_dist.items():
            percentage = count / max(1, statistics.get('total_objects', 1)) * 100
            report_text += f"  - {class_name}: {count} ({percentage:.1f}%)\n"
        report_text += "\n"
    
    # Add individual detections
    detections = results.get('detections', [])
    if detections:
        report_text += f"Individual Detections ({len(detections)}):\n"
        for i, det in enumerate(detections, 1):
            report_text += f"\n{i}. {det['class_name']}:\n"
            report_text += f"   - Confidence: {det['confidence']:.2f}\n"
            report_text += f"   - Bounding Box: {det['bbox']}\n"
            report_text += f"   - Area: {det['area']:,} px²\n"
            report_text += f"   - Center: {det['center']}\n"
            if det.get('has_mask', False):
                report_text += f"   - Mask Area: {det.get('mask_area', 0):,} px\n"
    
    return report_text

def display_obstacle_overview(results):
    """Display obstacle detection overview"""
    st.markdown("### Overview")
    
    if not results.get('success', False):
        st.error(f"❌ Detection failed: {results.get('error', 'Unknown error')}")
        return
    
    # Display detection image
    visualizations = results.get('visualizations', {})
    detections = results.get('detections', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Image")
        if 'original' in visualizations:
            st.image(f"data:image/png;base64,{visualizations['original']}", 
                    use_column_width=True)
    
    with col2:
        st.markdown("#### Detected Obstacles")
        if 'detection' in visualizations:
            st.image(f"data:image/png;base64,{visualizations['detection']}", 
                    use_column_width=True)
            st.caption(f"Detected {len(detections)} obstacles")
    
    # Show mask overlay if available
    if 'mask_overlay' in visualizations and visualizations['mask_overlay']:
        st.markdown("#### Segmentation Masks")
        st.image(f"data:image/png;base64,{visualizations['mask_overlay']}", 
                use_column_width=True)
        st.caption("Colored segmentation masks for detected obstacles")

def display_obstacle_visualizations(results):
    """Display obstacle detection visualizations"""
    st.markdown("### Detailed Visualizations")
    
    if not results.get('success', False):
        st.error(f"❌ Detection failed: {results.get('error', 'Unknown error')}")
        return
    
    visualizations = results.get('visualizations', {})
    detections = results.get('detections', [])
    
    # Display all available visualizations
    col1, col2 = st.columns(2)
    
    viz_items = [
        ('Original Image', 'original'),
        ('Detected Obstacles', 'detection'),
    ]
    
    if 'mask_overlay' in visualizations and visualizations['mask_overlay']:
        viz_items.append(('Segmentation Masks', 'mask_overlay'))
    
    for idx, (title, key) in enumerate(viz_items):
        col = col1 if idx % 2 == 0 else col2
        with col:
            with st.expander(f"🔍 {title}", expanded=True):
                if key in visualizations and visualizations[key]:
                    st.image(f"data:image/png;base64,{visualizations[key]}", 
                            use_column_width=True)
                    st.caption(title)
    
    # Show detection details
    if detections:
        st.markdown("#### Detection Details")
        for i, det in enumerate(detections):
            with st.expander(f"🚧 {det['class_name']} (Confidence: {det['confidence']:.2f})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bounding Box", f"{det['bbox'][2] - det['bbox'][0]}×{det['bbox'][3] - det['bbox'][1]}")
                with col2:
                    st.metric("Area", f"{det['area']:,} px²")
                with col3:
                    st.metric("Center", f"{det['center'][0]}, {det['center'][1]}")
                
                if det.get('has_mask', False) and det.get('mask_area', 0) > 0:
                    st.metric("Mask Area", f"{det['mask_area']:,} px")

def display_obstacle_statistics(results):
    """Display obstacle detection statistics"""
    st.markdown("### Detailed Statistics")
    
    if not results.get('success', False):
        st.error(f"❌ Detection failed: {results.get('error', 'Unknown error')}")
        return
    
    statistics = results.get('statistics', {})
    detections = results.get('detections', [])
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Objects", 
                 statistics.get('total_objects', 0),
                 delta="detections")
    with col2:
        st.metric("Average Confidence", 
                 f"{statistics.get('average_confidence', 0):.2f}",
                 delta="confidence")
    with col3:
        st.metric("Average Area", 
                 f"{statistics.get('average_area', 0):,.0f}",
                 delta="pixels²")
    with col4:
        mask_area = statistics.get('total_mask_area', 0)
        if mask_area > 0:
            st.metric("Total Mask Area", 
                     f"{mask_area:,}",
                     delta="pixels")
        else:
            st.metric("Segmentation", "Disabled")
    
    # Class distribution
    st.markdown("#### Class Distribution")
    class_dist = statistics.get('class_distribution', {})
    
    if class_dist:
        # Create bar chart for class distribution
        import pandas as pd
        df_classes = pd.DataFrame({
            'Class': list(class_dist.keys()),
            'Count': list(class_dist.values())
        })
        df_classes = df_classes.sort_values('Count', ascending=False)
        
        # Display as bar chart
        st.bar_chart(df_classes.set_index('Class'))
        
        # Display as table
        with st.expander("View Class Distribution Table"):
            for class_name, count in class_dist.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(min(count / max(1, len(detections)), 1.0), 
                               text=f"{class_name}:")
                with col2:
                    st.write(f"**{count}** ({count/max(1, len(detections))*100:.1f}%)")
    else:
        st.info("No class distribution data available")
    
    # Detection details table
    if detections:
        st.markdown("#### Detection Details")
        
        # Create a DataFrame for better display
        import pandas as pd
        df_detections = pd.DataFrame(detections)
        
        # Select and rename columns for display
        display_cols = ['class_name', 'confidence', 'area', 'center']
        if 'mask_area' in df_detections.columns:
            display_cols.append('mask_area')
        
        df_display = df_detections[display_cols].copy()
        df_display.columns = ['Class', 'Confidence', 'Area (px²)', 'Center (x,y)', 'Mask Area (px)'] if 'mask_area' in display_cols else ['Class', 'Confidence', 'Area (px²)', 'Center (x,y)']
        
        # Format numeric columns
        df_display['Confidence'] = df_display['Confidence'].apply(lambda x: f"{x:.2f}")
        df_display['Area (px²)'] = df_display['Area (px²)'].apply(lambda x: f"{x:,.0f}")
        if 'Mask Area (px)' in df_display.columns:
            df_display['Mask Area (px)'] = df_display['Mask Area (px)'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(df_display, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.markdown("#### 📥 Export Results")
    
    # Create report text
    # Note: image_source is not available here - will be passed from parent function
    report_text = "Obstacle Detection Results Report\n"
    report_text += "=" * 50 + "\n"
    report_text += f"Total Detections: {len(detections)}\n"
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name="obstacle_detection_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_dl2:
        if st.button("🖼️ Save Visualizations", use_container_width=True):
            st.info("Visualization save feature would be implemented here")
    
    with col_dl3:
        if st.button("🔄 New Detection", use_container_width=True):
            reset_obstacle_detection()
            st.rerun()

def display_obstacle_results():
    """Display obstacle detection results"""
    results = st.session_state.detection_results
    
    st.markdown("---")
    st.markdown('<h2 class="main-header">🚧 Obstacle Detection Results</h2>', unsafe_allow_html=True)
    
    # Use tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🖼️ Visualizations", "📈 Statistics"])
    
    with tab1:
        display_obstacle_overview(results)
    
    with tab2:
        display_obstacle_visualizations(results)
    
    with tab3:
        display_obstacle_statistics(results)

# ========== MAIN APP ==========

# App title
st.markdown('<h1 class="main-header">🚗 Advanced Lane Detection System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=100)
    st.markdown("### Settings")
    
    # Mode selection
    mode = st.radio(
        "Select Detection Mode:",
        ["straight", "curved", "obstacle"],
        index=0,
        help="Choose between straight, curved lane detection, or obstacle detection",
        on_change=lambda: [reset_processing(), reset_obstacle_detection()]
    )
    
    # Update current mode
    if mode != st.session_state.current_mode:
        st.session_state.current_mode = mode
        reset_processing()
        reset_obstacle_detection()
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a road for detection",
        on_change=lambda: [reset_processing(), reset_obstacle_detection()]
    )
    
    # Sample images section
    st.markdown("### Sample Images")
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("🚦 Straight Lane", use_container_width=True, key="sample_straight"):
            st.session_state.use_sample = True
            st.session_state.sample_mode = "straight"
            st.session_state.current_image = create_sample_image("straight")
            reset_processing()
            reset_obstacle_detection()
            st.rerun()
    
    with sample_col2:
        if st.button("🔄 Curved Lane", use_container_width=True, key="sample_curved"):
            st.session_state.use_sample = True
            st.session_state.sample_mode = "curved"
            st.session_state.current_image = create_sample_image("curved")
            reset_processing()
            reset_obstacle_detection()
            st.rerun()
    
    # Obstacle detection settings (only show in obstacle mode)
    if mode == "obstacle":
        st.markdown("---")
        st.markdown("### Obstacle Detection Settings")
        
        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Higher values = fewer but more confident detections"
        )
        
        # Segmentation toggle
        segmentation = st.toggle(
            "Enable Segmentation Masks",
            value=True,
            help="Show colored masks over detected obstacles"
        )
        
        # Mask transparency
        mask_alpha = st.slider(
            "Mask Transparency",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="How transparent the segmentation masks are"
        )
        
        # Initialize detector button
        if st.button("🔄 Initialize Obstacle Detector", use_container_width=True):
            init_obstacle_detector()
    
    # Clear button
    if st.session_state.use_sample or uploaded_file is not None:
        if st.button("🗑️ Clear Image", type="secondary", use_container_width=True):
            st.session_state.use_sample = False
            st.session_state.sample_mode = None
            st.session_state.current_image = None
            st.session_state.processing_results = None
            st.session_state.detection_results = None
            st.session_state.image_processed = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    with st.expander("ℹ️ Learn more about the system"):
        st.markdown("""
        **Features:**
        - **Straight Lane Detection**: Hough Transform for line detection
        - **Curved Lane Detection**: Edge analysis with multiple visualizations
        - **Obstacle Detection**: YOLOv8 with segmentation masks for road obstacles
        - **Color Filtering**: White and yellow lane detection
        - **ROI Masking**: Focus on relevant road area
        
        **Processing Steps:**
        1. Region of Interest (ROI) selection
        2. Color filtering (white/yellow)
        3. Edge detection (Canny/Sobel)
        4. Line detection (Hough transform)
        5. Obstacle detection with YOLO
        6. Result visualization with segmentation masks
        """)

# Main content area
if uploaded_file is not None or st.session_state.use_sample:
    # Determine image source
    if uploaded_file is not None:
        # Using uploaded file
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        image_source = "Uploaded Image"
        st.session_state.current_image = image_array
        st.session_state.use_sample = False
    else:
        # Using sample
        image_array = st.session_state.current_image
        image_source = f"Sample {st.session_state.sample_mode.capitalize()} Lane"
    
    # Store current mode
    current_mode = st.session_state.current_mode
    
    # Display image information
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<h3 class="sub-header">📷 Input Image</h3>', unsafe_allow_html=True)
        st.image(image_array, caption=image_source, use_column_width=True)
        
        # Image info
        with st.expander("📊 Image Details"):
            st.write(f"**Dimensions:** {image_array.shape[1]} × {image_array.shape[0]} pixels")
            st.write(f"**Channels:** {image_array.shape[2] if len(image_array.shape) == 3 else 1}")
            st.write(f"**Data Type:** {image_array.dtype}")
            st.write(f"**Detection Mode:** {current_mode.capitalize()}")
            if st.session_state.use_sample:
                st.warning("⚠️ Using sample image. Upload your own for real-world detection.")
    
    with col2:
        st.markdown('<h3 class="sub-header">⚙️ Processing Controls</h3>', unsafe_allow_html=True)
        
        if current_mode in ['straight', 'curved']:
            # Lane detection controls
            process_col1, process_col2 = st.columns([3, 1])
            with process_col1:
                if st.button("🔍 Process Lane Detection", type="primary", use_container_width=True, 
                            disabled=st.session_state.image_processed):
                    with st.spinner(f"Processing {current_mode} lane detection..."):
                        try:
                            results = process_image(image_array, current_mode)
                            st.session_state.processing_results = results
                            st.session_state.image_processed = True
                            st.success("✅ Lane detection completed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
            
            with process_col2:
                if st.button("🔄 Reset", type="secondary", use_container_width=True):
                    reset_processing()
                    st.rerun()
            
            # Show processing status
            if st.session_state.image_processed:
                st.balloons()
                st.info("✅ Image has been processed. View results in the Results tab.")
            else:
                st.info("👆 Click 'Process Lane Detection' to start detection")
            
            # Quick stats if processed
            if st.session_state.processing_results:
                st.markdown("### 📈 Quick Statistics")
                if current_mode == 'straight':
                    st.metric("Lines Detected", st.session_state.processing_results.get('lines_detected', 0))
                else:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Edges", f"{st.session_state.processing_results.get('edges_count', 0):,}")
                    with col_stat2:
                        st.metric("Density", f"{st.session_state.processing_results.get('edge_density', 0):.1f}%")
        
        else:  # Obstacle detection mode
            # Initialize detector if needed
            if st.session_state.obstacle_detector is None:
                init_obstacle_detector()
            
            # Obstacle detection controls
            process_col1, process_col2 = st.columns([3, 1])
            with process_col1:
                if st.button("🔍 Detect Obstacles", type="primary", use_container_width=True):
                    if st.session_state.obstacle_detector:
                        with st.spinner("Detecting obstacles with segmentation..."):
                            try:
                                # Convert PIL image to OpenCV format
                                if isinstance(image_array, np.ndarray):
                                    # Image is already in numpy array
                                    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if len(image_array.shape) == 3 else image_array
                                else:
                                    # Convert PIL to numpy
                                    img_rgb = np.array(image_array)
                                    if len(img_rgb.shape) == 3:
                                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                                
                                # Run obstacle detection
                                results = st.session_state.obstacle_detector.detect(
                                    image=img_rgb,
                                    confidence=confidence,
                                    show_masks=segmentation,
                                    mask_alpha=mask_alpha
                                )
                                
                                st.session_state.detection_results = results
                                if results.get('success', False):
                                    st.success(f"✅ Detected {results.get('total_detections', 0)} obstacles!")
                                else:
                                    st.warning(f"⚠️ Detection completed with issues: {results.get('error', 'Unknown error')}")
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Obstacle detection failed: {str(e)}")
                    else:
                        st.error("❌ Obstacle detector not initialized. Please initialize from sidebar.")
            
            with process_col2:
                if st.button("🔄 Reset", type="secondary", use_container_width=True):
                    reset_obstacle_detection()
                    st.rerun()
            
            # Show detection status
            if st.session_state.detection_results:
                st.balloons()
                if st.session_state.detection_results.get('is_mock', False):
                    st.warning("⚠️ Showing mock detections (real model not available)")
                st.info("✅ Obstacle detection completed. View results in the Results tab.")
            else:
                if st.session_state.obstacle_detector:
                    st.info("👆 Click 'Detect Obstacles' to start obstacle detection")
                else:
                    st.warning("⚠️ Obstacle detector not ready. Click 'Initialize Obstacle Detector' in sidebar.")
            
            # Quick stats if processed
            if st.session_state.detection_results and st.session_state.detection_results.get('success', False):
                st.markdown("### 📈 Quick Statistics")
                stats = st.session_state.detection_results.get('statistics', {})
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Objects Detected", stats.get('total_objects', 0))
                with col_stat2:
                    st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.2f}")
                with col_stat3:
                    mask_area = stats.get('total_mask_area', 0)
                    if mask_area > 0:
                        st.metric("Mask Area", f"{mask_area:,} px")

    # Results Section - Using Tabs for separate window effect
    if st.session_state.processing_results and current_mode in ['straight', 'curved']:
        from utils import display_results_section
        display_results_section(current_mode, image_source, image_array)
    
    elif st.session_state.detection_results and current_mode == 'obstacle':
        display_obstacle_results()

else:
    # Welcome screen
    from utils import display_welcome_screen
    display_welcome_screen()

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🚗 **Advanced Lane Detection System** | "
        "Computer Vision Project | "
        "Built with Streamlit, OpenCV & YOLO"
        "</div>",
        unsafe_allow_html=True
    ) 