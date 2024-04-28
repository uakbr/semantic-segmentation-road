import numpy as np
import cv2
from PIL import Image

def visualize_segmentation(segmentation_map, input_image, alpha=0.5):
    """
    Visualizes segmentation map on input image.
    
    Args:
        segmentation_map (numpy.ndarray): Segmentation map of shape (height, width).
        input_image (numpy.ndarray): Input image of shape (height, width, 3).
        alpha (float): Opacity of segmentation map. Default: 0.5.
        
    Returns:
        numpy.ndarray: Visualization image of shape (height, width, 3).
    """
    # Define color map
    color_map = {
        0: [0, 0, 0],        # Void
        1: [128, 64, 128],   # Road
        2: [244, 35, 232],   # Sidewalk
        3: [70, 70, 70],     # Building
        4: [102, 102, 156],  # Wall
        5: [190, 153, 153],  # Fence
        6: [153, 153, 153],  # Pole
        7: [250, 170, 30],   # Traffic Light
        8: [220, 220, 0],    # Traffic Sign
        9: [107, 142, 35],   # Vegetation 
        10: [152, 251, 152], # Terrain
        11: [70, 130, 180],  # Sky
        12: [220, 20, 60],   # Person
        13: [255, 0, 0],     # Rider
        14: [0, 0, 142],     # Car
        15: [0, 0, 70],      # Truck
        16: [0, 60, 100],    # Bus
        17: [0, 80, 100],    # Train
        18: [0, 0, 230],     # Motorcycle
        19: [119, 11, 32]    # Bicycle
    }
    
    # Create RGB segmentation map
    rgb_seg_map = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        rgb_seg_map[segmentation_map == label] = color
        
    # Convert input image to RGB if grayscale
    if len(input_image.shape) < 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        
    # Resize segmentation map to match input image size
    rgb_seg_map = cv2.resize(rgb_seg_map, (input_image.shape[1], input_image.shape[0]))
    
    # Blend segmentation map and input image
    blended = cv2.addWeighted(input_image, 1 - alpha, rgb_seg_map, alpha, 0)
    
    return blended


def visualize_video(video_path, segmentation_maps, output_path, fps=30):
    """
    Visualizes segmentation maps on video.
    
    Args:
        video_path (str): Path to input video file.
        segmentation_maps (list): List of segmentation maps, one for each frame.
        output_path (str): Path to save output video file.
        fps (int): Frames per second for output video. Default: 30.
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Visualize segmentation maps on each frame
    for seg_map in segmentation_maps:
        ret, frame = cap.read()
        if not ret:
            break
            
        blended_frame = visualize_segmentation(seg_map, frame)
        out.write(blended_frame)
        
    cap.release()
    out.release()
    
    print(f'Visualization complete. Output saved to: {output_path}')