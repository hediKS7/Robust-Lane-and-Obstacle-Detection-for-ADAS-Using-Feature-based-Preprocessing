import streamlit as st
import numpy as np
import cv2

def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown("""
    <style>
    
    /* ===== HEADERS ===== */
    .main-header {
        font-size: 2.6rem;
        color: #64B5F6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.6rem;
        color: #90CAF9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* ===== INFO BOX ===== */
    .info-box {
        background: linear-gradient(135deg, #1E1E2E, #23233A);
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 4px solid #64B5F6;
        color: #E0E0E0;
    }
    
    /* ===== STATS BOX ===== */
    .stats-box {
        background-color: #1F2933;
        padding: 0.9rem;
        border-radius: 0.75rem;
        border-left: 4px solid #64B5F6;
        margin: 0.6rem 0;
        color: #E0E0E0;
    }
    
    /* ===== RESULT TAB ===== */
    .result-tab {
        background-color: #1B1F2A;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #2E3440;
        color: #EAEAEA;
    }
    
    /* ===== BUTTONS ===== */
    .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        color: white;
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.25s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #42A5F5, #64B5F6);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(66, 165, 245, 0.35);
    }
    
    /* Obstacle detection specific button */
    .obstacle-button button {
        background: linear-gradient(90deg, #FF9800, #FF5722);
    }
    
    .obstacle-button button:hover {
        background: linear-gradient(90deg, #FF5722, #E64A19);
        box-shadow: 0 8px 20px rgba(255, 87, 34, 0.35);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #111827;
        padding: 0.6rem;
        border-radius: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #1F2933;
        color: #CBD5E1;
        border-radius: 0.75rem 0.75rem 0 0;
        gap: 1rem;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        font-weight: 500;
    }
    
    /* Active tab */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        color: white;
    }
    
    /* ===== SCROLLBAR (Optional) ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    
    </style>
    """, unsafe_allow_html=True)

def reset_processing():
    """Reset processing state"""
    st.session_state.processing_results = None
    st.session_state.image_processed = False

def create_sample_image(mode):
    """Create sample image based on mode"""
    if mode == "straight":
        # Create a sample straight lane image
        image_array = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw a simple road with lane markings
        cv2.rectangle(image_array, (0, 200), (600, 400), (100, 100, 100), -1)  # Road
        cv2.line(image_array, (300, 200), (300, 400), (255, 255, 255), 10)  # Center line
        cv2.line(image_array, (100, 200), (100, 400), (255, 255, 255), 5)  # Left lane
        cv2.line(image_array, (500, 200), (500, 400), (255, 255, 255), 5)  # Right lane
        # Add some yellow lane
        cv2.line(image_array, (80, 200), (80, 400), (255, 255, 0), 5)  # Yellow lane
        
    elif mode == "curved":
        # Create a sample curved lane image
        image_array = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw a curved road
        cv2.rectangle(image_array, (0, 150), (600, 400), (100, 100, 100), -1)  # Road
        
        # Draw curved lane markings
        pts_left = np.array([[200, 150], [150, 250], [200, 350], [250, 400]], np.int32)
        pts_right = np.array([[400, 150], [450, 250], [400, 350], [350, 400]], np.int32)
        cv2.polylines(image_array, [pts_left], False, (255, 255, 255), 5)
        cv2.polylines(image_array, [pts_right], False, (255, 255, 0), 5)  # Yellow curve
    
    else:  # obstacle mode
        # Create a sample obstacle image
        image_array = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw a road
        cv2.rectangle(image_array, (0, 150), (600, 400), (100, 100, 100), -1)  # Road
        
        # Draw some obstacles
        cv2.rectangle(image_array, (100, 200), (200, 300), (0, 0, 255), -1)  # Blue car
        cv2.circle(image_array, (400, 250), 30, (0, 255, 255), -1)  # Yellow obstacle
        cv2.rectangle(image_array, (500, 180), (550, 220), (255, 0, 0), -1)  # Red obstacle
        
        # Add some lane markings for context
        cv2.line(image_array, (300, 150), (300, 400), (255, 255, 255), 5)  # Lane marking
        
    return image_array

def display_results_section(current_mode, image_source, image_array):
    """Display the results section with tabs"""
    st.markdown("---")
    st.markdown('<h2 class="main-header">📊 Processing Results</h2>', unsafe_allow_html=True)
    
    # Use tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🖼️ Visualizations", "📈 Statistics"])
    
    with tab1:
        display_overview_tab(current_mode)
    
    with tab2:
        display_visualizations_tab(current_mode)
    
    with tab3:
        display_statistics_tab(current_mode, image_source, image_array)

def display_overview_tab(current_mode):
    """Display overview tab content"""
    st.markdown("### Overview")
    
    if current_mode == 'straight':
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Original Image with ROI")
            st.image(f"data:image/png;base64,{st.session_state.processing_results['original_with_roi']}", 
                    use_column_width=True)
            st.markdown("#### Edge Detection")
            st.image(f"data:image/png;base64,{st.session_state.processing_results['edges_roi']}", 
                    use_column_width=True)
        
        with cols[1]:
            st.markdown("#### Color Filtered")
            st.image(f"data:image/png;base64,{st.session_state.processing_results['color_selected']}", 
                    use_column_width=True)
            st.markdown("#### Lane Detection")
            st.image(f"data:image/png;base64,{st.session_state.processing_results['final_detection']}", 
                    use_column_width=True)
    
    else:  # curved mode
        st.markdown("#### Best Visualization")
        st.image(f"data:image/png;base64,{st.session_state.processing_results['single_visualization']}", 
                use_column_width=True)
        
        st.markdown("#### Grid Visualization")
        st.image(f"data:image/png;base64,{st.session_state.processing_results['grid_visualization']}", 
                use_column_width=True)

def display_visualizations_tab(current_mode):
    """Display visualizations tab content"""
    st.markdown("### Detailed Visualizations")
    
    if current_mode == 'straight':
        # For straight lanes, show the 4 steps
        steps = [
            ("Original + ROI", "original_with_roi"),
            ("Color Filtered", "color_selected"),
            ("Edge Detection", "edges_roi"),
            ("Lane Detection", "final_detection")
        ]
        
        cols = st.columns(2)
        for idx, (title, key) in enumerate(steps):
            with cols[idx % 2]:
                with st.expander(f"🔍 {title}", expanded=True):
                    st.image(f"data:image/png;base64,{st.session_state.processing_results[key]}", 
                            use_column_width=True)
                    st.caption(f"Step {idx + 1}: {title}")
    
    else:  # curved mode
        # Show individual images in expanders
        individual_images = st.session_state.processing_results['individual_images']
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        image_titles = {
            'edges_roi': "Canny Edges in ROI",
            'overlay_white': "White Edges Overlay",
            'overlay_red': "Red Edges Overlay",
            'overlay_green': "Green Edges Overlay",
            'edges_visualization': "Edge Visualization",
            'roi_outline': "ROI Outline"
        }
        
        for idx, (key, img_base64) in enumerate(individual_images.items()):
            col = col1 if idx % 2 == 0 else col2
            with col:
                with st.expander(f"📸 {image_titles.get(key, key)}", expanded=True):
                    st.image(f"data:image/png;base64,{img_base64}", 
                            use_column_width=True)
                    st.caption(image_titles.get(key, key))

def display_statistics_tab(current_mode, image_source, image_array):
    """Display statistics tab content"""
    st.markdown("### Detailed Statistics")
    
    if current_mode == 'straight':
        # Create metrics cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lines Detected", 
                     st.session_state.processing_results.get('lines_detected', 0),
                     delta="lines")
        with col2:
            st.metric("Detection Mode", "Straight Lanes")
        with col3:
            st.metric("Processing Status", "Completed", delta="success")
        
        # Additional info
        st.markdown("#### Processing Information")
        st.info("""
        **Straight Lane Detection Process:**
        1. ROI selection using trapezoidal mask
        2. Color filtering for white/yellow lanes
        3. Canny edge detection
        4. Hough transform for line detection
        """)
    
    else:  # curved mode
        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Edges", 
                     f"{st.session_state.processing_results.get('edges_count', 0):,}",
                     delta="pixels")
        with col2:
            st.metric("ROI Area", 
                     f"{st.session_state.processing_results.get('roi_area', 0):,}",
                     delta="pixels")
        with col3:
            st.metric("Edge Density", 
                     f"{st.session_state.processing_results.get('edge_density', 0):.2f}%",
                     delta="%")
        with col4:
            st.metric("Detection Mode", "Curved Lanes")
        
        # Edge density visualization
        st.markdown("#### Edge Density Analysis")
        edge_density = st.session_state.processing_results.get('edge_density', 0)
        
        # Create a progress bar for edge density
        st.progress(min(edge_density / 100, 1.0), 
                   text=f"Edge Density: {edge_density:.2f}%")
        
        if edge_density < 5:
            st.info("Low edge density - smooth road or few lane markings")
        elif edge_density < 15:
            st.success("Moderate edge density - good lane visibility")
        else:
            st.warning("High edge density - complex road or many edges detected")
    
    # Download section
    st.markdown("---")
    st.markdown("#### 📥 Export Results")
    
    # Create report text
    report_text = create_report_text(current_mode, image_source, image_array)
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name=f"lane_detection_{current_mode}_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_dl2:
        if st.button("🖼️ Save Visualizations", use_container_width=True):
            st.info("Visualization save feature would be implemented here")
    
    with col_dl3:
        if st.button("🔄 New Processing", use_container_width=True):
            reset_processing()
            st.rerun()

def create_report_text(current_mode, image_source, image_array):
    """Create report text for download"""
    report_text = f"Lane Detection Results Report\n"
    report_text += "=" * 40 + "\n"
    report_text += f"Detection Mode: {current_mode.capitalize()}\n"
    report_text += f"Image Source: {image_source}\n"
    report_text += f"Image Dimensions: {image_array.shape[1]} × {image_array.shape[0]}\n"
    report_text += f"Processing Timestamp: {st.session_state.get('processing_time', 'N/A')}\n\n"
    
    if current_mode == 'straight':
        report_text += f"Lines Detected: {st.session_state.processing_results.get('lines_detected', 0)}\n"
    else:
        report_text += f"Edges Detected: {st.session_state.processing_results.get('edges_count', 0):,}\n"
        report_text += f"ROI Area: {st.session_state.processing_results.get('roi_area', 0):,}\n"
        report_text += f"Edge Density: {st.session_state.processing_results.get('edge_density', 0):.2f}%\n"
    
    return report_text

def display_welcome_screen():
    """Display welcome screen when no image is loaded"""
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to the Advanced Lane & Obstacle Detection System</h3>
    <p>This advanced system uses computer vision techniques to detect lane markings and obstacles in road images. 
    Choose your detection mode and upload an image to get started!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown('<h3 class="sub-header">🎯 Key Features</h3>', unsafe_allow_html=True)
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        <div class="stats-box">
        <h4>🛣️ Straight Lane Detection</h4>
        <ul>
        <li>Hough Transform for precise line detection</li>
        <li>Color filtering for white/yellow lanes</li>
        <li>ROI masking for focused analysis</li>
        <li>Real-time processing and visualization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-box">
        <h4>🚧 Obstacle Detection</h4>
        <ul>
        <li>YOLOv8 deep learning model</li>
        <li>Segmentation masks for precise outlines</li>
        <li>Multiple obstacle classes detection</li>
        <li>Confidence-based filtering</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="stats-box">
        <h4>🔄 Curved Lane Detection</h4>
        <ul>
        <li>Advanced edge detection with Sobel+Canny</li>
        <li>Multiple visualization methods</li>
        <li>Edge density analysis</li>
        <li>Detailed statistics and metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stats-box">
        <h4>🚀 Advanced Features</h4>
        <ul>
        <li>Tab-based results organization</li>
        <li>Session state management</li>
        <li>Real-time progress tracking</li>
        <li>Export capabilities</li>
        <li>Responsive design</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample outputs
    st.markdown('<h3 class="sub-header">🖼️ Sample Outputs Preview</h3>', unsafe_allow_html=True)
    
    sample_tab1, sample_tab2, sample_tab3 = st.tabs(["Straight Lane Demo", "Curved Lane Demo", "Obstacle Detection"])
    
    with sample_tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Create sample straight visualization
            straight_demo = create_sample_image("straight")
            st.image(straight_demo, caption="Sample Straight Lane Input", use_column_width=True)
        with col2:
            st.markdown("""
            **Expected Outputs:**
            - ROI trapezoid overlay
            - Color-filtered lanes
            - Edge detection results
            - Detected lines with Hough transform
            
            **Use Cases:**
            - Highway lane detection
            - Straight road analysis
            - Lane keeping systems
            """)
    
    with sample_tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Create sample curved visualization
            curved_demo = create_sample_image("curved")
            st.image(curved_demo, caption="Sample Curved Lane Input", use_column_width=True)
        with col2:
            st.markdown("""
            **Expected Outputs:**
            - Multiple edge visualizations
            - Edge density analysis
            - ROI analysis
            - Curvature estimation
            
            **Use Cases:**
            - Curved road analysis
            - Exit ramp detection
            - Curvature measurement
            """)
    
    with sample_tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Create sample obstacle visualization
            obstacle_demo = create_sample_image("obstacle")
            st.image(obstacle_demo, caption="Sample Obstacle Detection Input", use_column_width=True)
        with col2:
            st.markdown("""
            **Expected Outputs:**
            - Bounding boxes around obstacles
            - Segmentation masks (colored overlays)
            - Confidence scores
            - Class labels
            - Statistics and metrics
            
            **Use Cases:**
            - Autonomous driving
            - ADAS systems
            - Road safety monitoring
            - Traffic analysis
            """)
    
    # Quick start guide
    st.markdown("---")
    st.markdown('<h3 class="sub-header">🚀 Quick Start Guide</h3>', unsafe_allow_html=True)
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        ### 1. Select Mode
        Choose between straight, curved, or obstacle detection based on your needs.
        
        **Tip:** Use obstacle mode for detecting cars, pedestrians, etc.
        """)
    
    with guide_col2:
        st.markdown("""
        ### 2. Upload Image
        Upload a clear road image or use our sample images for testing.
        
        **Tip:** Ensure good lighting and minimal obstructions.
        """)
    
    with guide_col3:
        st.markdown("""
        ### 3. Process & Analyze
        Click process and explore results in the organized tabs.
        
        **Tip:** For obstacle detection, adjust confidence threshold in sidebar.
        """)