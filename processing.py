import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image
import io
import base64

def process_straight_lane(image_path):
    """
    Process image for straight lane detection
    Returns: dict containing processed images and info
    """
    # ================================================
    # 1) READ IMAGE
    # ================================================
    if isinstance(image_path, str):
        image = mpimg.imread(image_path)
    else:
        # If it's already an image array
        image = image_path
    
    original = np.copy(image)
    ysize, xsize = image.shape[0], image.shape[1]
    
    # ================================================
    # 2) TRAPEZOID ROI 
    # ================================================
    left_bottom_x = 0
    right_bottom_x = int(xsize)
    left_top_x = int(xsize * 0.4)
    right_top_x = int(xsize * 0.6)
    top_y = int(ysize * 0.6)
    
    vertices = np.array([[
        (left_bottom_x, ysize),
        (right_bottom_x, ysize),
        (right_top_x, top_y),
        (left_top_x, top_y)
    ]], dtype=np.int32)
    
    mask = np.zeros((ysize, xsize), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    
    # ================================================
    # 3) COLOR FILTER — DETECT WHITE + YELLOW LANES
    # ================================================
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # White mask
    white_mask = (v > 200) & (s < 60)
    
    # Yellow mask
    yellow_mask = ((h > 15) & (h < 35)) & (s > 80) & (v > 80)
    
    # Combined mask
    color_mask = (white_mask | yellow_mask)
    color_mask &= (mask > 0)   # apply ROI
    
    color_selected = np.copy(image)
    color_selected[~color_mask] = [0, 0, 0]
    
    # ================================================
    # 4) CANNY EDGE DETECTION
    # ================================================
    gray = cv2.cvtColor(color_selected, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Apply ROI on edges
    edges_roi = cv2.bitwise_and(edges, edges, mask=mask)
    
    # ================================================
    # 5) HOUGH LINES
    # ================================================
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )
    
    line_image = np.copy(image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Create ROI outline image - FIXED THE TYPO HERE
    roi_outline = np.copy(original)  # Changed from rooi_outline to roi_outline
    cv2.polylines(roi_outline, vertices, True, (255, 0, 0), 2)
    
    # Convert images to base64 for display
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str
    
    # Create figures
    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.imshow(original)
    x = [p[0] for p in vertices[0]] + [vertices[0][0][0]]
    y = [p[1] for p in vertices[0]] + [vertices[0][0][1]]
    ax1.plot(x, y, 'r--', lw=2)
    ax1.set_title("1. Original Image + ROI")
    ax1.axis('off')
    img1 = fig_to_base64(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(7,5))
    ax2.imshow(color_selected)
    ax2.set_title("2. White + Yellow Lane Pixels Selected")
    ax2.axis('off')
    img2 = fig_to_base64(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(7,5))
    ax3.imshow(edges_roi, cmap='gray')
    ax3.set_title("3. Canny Edge Detection inside ROI")
    ax3.axis('off')
    img3 = fig_to_base64(fig3)
    
    fig4, ax4 = plt.subplots(figsize=(7,5))
    ax4.imshow(line_image)
    ax4.set_title("4. Final Lane Line Detection (Hough)")
    ax4.axis('off')
    img4 = fig_to_base64(fig4)
    
    return {
        'original_with_roi': img1,
        'color_selected': img2,
        'edges_roi': img3,
        'final_detection': img4,
        'lines_detected': len(lines) if lines is not None else 0
    }


def process_curved_lane(image_path):
    """
    Process image for curved lane detection
    Returns: dict containing processed images and info
    """
    # ============================
    # PARAMETERS
    # ============================
    nwindows = 200
    margin = 20
    minpix = 150
    sobel_ksize = 3
    
    # ================================================
    # 1) READ IMAGE
    # ================================================
    if isinstance(image_path, str):
        image = mpimg.imread(image_path)
    else:
        image = image_path
    
    original = np.copy(image)
    ysize, xsize = image.shape[:2]
    
    # ================================================
    # 2) TRAPEZOID ROI
    # ================================================
    left_bottom_x = 0
    right_bottom_x = int(xsize)
    left_top_x = int(xsize * 0.4)
    right_top_x = int(xsize * 0.6)
    top_y = int(ysize * 0.80)
    
    vertices = np.array([[
        (left_bottom_x, ysize),
        (right_bottom_x, ysize),
        (right_top_x, top_y),
        (left_top_x, top_y)
    ]], dtype=np.int32)
    
    mask = np.zeros((ysize, xsize), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    
    roi_img = np.copy(image)
    roi_img[mask == 0] = 0
    
    # ================================================
    # 3) COLOR FILTER
    # ================================================
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    white_mask = (v > 200) & (s < 60)
    yellow_mask = ((h > 15) & (h < 35)) & (s > 80) & (v > 80)
    
    color_mask = (white_mask | yellow_mask)
    color_mask &= (mask > 0)
    
    color_selected = np.copy(image)
    color_selected[~color_mask] = 0
    
    # ================================================
    # 4) EDGE DETECTION
    # ================================================
    gray = cv2.cvtColor(color_selected, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = (grad_mag / np.max(grad_mag) * 255).astype(np.uint8)
    
    edges = cv2.Canny(blur, 50, 150)
    edges_combined = np.zeros_like(edges)
    edges_combined[(edges > 0) | (grad_mag > 80)] = 255
    
    edges_roi = cv2.bitwise_and(edges_combined, edges_combined, mask=mask)
    
    # ================================================
    # 5) CREATE IMAGE WITH CANNY EDGES ONLY ON ROI
    # ================================================
    # Create edges from Canny only (not combined with Sobel)
    edges_canny = cv2.Canny(blur, 50, 150)
    
    # Apply mask to get only edges within ROI
    edges_canny_roi = cv2.bitwise_and(edges_canny, edges_canny, mask=mask)
    
    # Create overlay of edges on original image
    # Method 1: White edges on original image
    overlay_white = np.copy(original)
    overlay_white[edges_canny_roi > 0] = [255, 255, 255]
    
    # Method 2: Red edges on original image (more visible)
    overlay_red = np.copy(original)
    red_edges = np.zeros_like(original)
    red_edges[edges_canny_roi > 0] = [255, 0, 0]
    overlay_red = cv2.addWeighted(original, 0.8, red_edges, 0.6, 0)
    
    # Method 3: Green edges (alternative)
    overlay_green = np.copy(original)
    green_edges = np.zeros_like(original)
    green_edges[edges_canny_roi > 0] = [0, 255, 0]
    overlay_green = cv2.addWeighted(original, 0.8, green_edges, 0.6, 0)
    
    # Create ROI outline image
    roi_outline = np.copy(original)
    cv2.polylines(roi_outline, vertices, True, (0, 255, 0), 2)
    
    # Create edges visualization
    edges_visualization = np.zeros_like(original)
    edges_visualization[edges_canny_roi > 0] = [255, 0, 0]
    
    # Convert images to base64
    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str
    
    def array_to_base64(img_array, title=""):
        fig, ax = plt.subplots(figsize=(7,5))
        ax.imshow(img_array)
        ax.set_title(title)
        ax.axis('off')
        return fig_to_base64(fig)
    
    # Create individual images
    img1 = array_to_base64(edges_canny_roi, "Canny Edges in ROI Only")
    img2 = array_to_base64(overlay_white, "Original + White Canny Edges")
    img3 = array_to_base64(overlay_red, "Original + Red Canny Edges")
    img4 = array_to_base64(overlay_green, "Original + Green Canny Edges")
    img5 = array_to_base64(edges_visualization, "Canny Edges Visualization")
    img6 = array_to_base64(roi_outline, "ROI Region Outline")
    
    # Create grid visualization
    fig_grid, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = [
        "Canny Edges in ROI Only",
        "Original + White Canny Edges",
        "Original + Red Canny Edges",
        "Original + Green Canny Edges",
        "Canny Edges Visualization",
        "ROI Region Outline"
    ]
    
    images = [
        edges_canny_roi,
        overlay_white,
        overlay_red,
        overlay_green,
        edges_visualization,
        roi_outline
    ]
    
    for ax, img, title in zip(axes.flat, images, titles):
        if len(img.shape) == 2:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    grid_img = fig_to_base64(fig_grid)
    
    # Create single best visualization
    fig_single, ax_single = plt.subplots(figsize=(10, 6))
    ax_single.imshow(overlay_red)
    ax_single.set_title("Original Image with Canny Edges in ROI (Red)")
    ax_single.axis('off')
    single_img = fig_to_base64(fig_single)
    
    # Calculate statistics
    edges_count = np.sum(edges_canny_roi > 0)
    roi_area = np.sum(mask > 0)
    edge_density = (edges_count / roi_area * 100) if roi_area > 0 else 0
    
    return {
        'grid_visualization': grid_img,
        'single_visualization': single_img,
        'edges_count': int(edges_count),
        'roi_area': int(roi_area),
        'edge_density': float(edge_density),
        'individual_images': {
            'edges_roi': img1,
            'overlay_white': img2,
            'overlay_red': img3,
            'overlay_green': img4,
            'edges_visualization': img5,
            'roi_outline': img6
        }
    }


def process_image(image_path, mode='straight'):
    """
    Main processing function
    mode: 'straight' or 'curved'
    """
    if mode == 'straight':
        return process_straight_lane(image_path)
    elif mode == 'curved':
        return process_curved_lane(image_path)
    else:
        raise ValueError("Mode must be 'straight' or 'curved'")