import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_path, output_path, processing_func, fps=30):
        """
        Initializes the VideoProcessor.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the processed video file.
            processing_func (callable): Function to process each frame of the video.
            fps (int): Frames per second of the output video. Default: 30.
        """
        self.video_path = video_path
        self.output_path = output_path
        self.processing_func = processing_func
        self.fps = fps

    def process_video(self):
        """
        Processes the input video by applying the processing function to each frame
        and saves the processed video to the specified output path.
        """
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.processing_func(frame)
            out.write(processed_frame)

        cap.release()
        out.release()

    @staticmethod
    def apply_segmentation_mask(frame, segmentation_mask, alpha=0.5, color_map=None):
        """
        Applies the segmentation mask to the input frame.

        Args:
            frame (numpy.ndarray): Input frame (H, W, 3).
            segmentation_mask (numpy.ndarray): Segmentation mask (H, W).
            alpha (float): Opacity of the segmentation mask. Default: 0.5.
            color_map (dict): Dictionary mapping class indices to colors. Default: None.

        Returns:
            numpy.ndarray: Frame with the segmentation mask applied (H, W, 3).
        """
        if color_map is None:
            color_map = {
                0: [0, 0, 0],        # Background
                1: [128, 0, 0],      # Class 1
                2: [0, 128, 0],      # Class 2
                3: [128, 128, 0],    # Class 3
                # Add more colors for additional classes
            }

        mask = np.zeros_like(frame)
        for class_idx, color in color_map.items():
            mask[segmentation_mask == class_idx] = color

        return cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)