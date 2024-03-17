import cv2
import numpy as np

def compute_mad(prev,curr):
    M, N = curr.shape
    return np.sum(np.abs(curr - prev)) / (M*N)

def estimate_motion_vectors(prev_frame, curr_frame, block_size=16, search_range=7):
    rows, cols = curr_frame.shape
    motion_vectors = []

    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):
            curr_block = curr_frame[row:row+block_size, col:col+block_size]
            best_mad = np.inf
            best_vector = (0, 0)

            for x in range(-search_range, search_range +1):
                for y in range(-search_range, search_range+1):
                    prev_row, prev_col = row + x, col + y
                    if 0 <= prev_row < rows - block_size and 0 <= prev_col < cols - block_size:
                        prev_block = prev_frame[prev_row:prev_row+block_size, prev_col:prev_col+block_size]

                        mad = compute_mad(prev_block, curr_block)
                        if mad < best_mad:
                            best_mad = mad
                            best_vector = (x, y)
            # assign just first pixel of block
            motion_vectors.append(((row, col), best_vector))

    return motion_vectors

def visualize_motion_vectors(curr_frame, motion_vectors, block_size=16):
    # Create a copy of the current frame to draw on
    vis_frame = curr_frame.copy()
    
    # Convert to color if the current frame is grayscale
    if len(vis_frame.shape) == 2 or vis_frame.shape[2] == 1:
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
    
    for (row, col), (dx, dy) in motion_vectors:
        # Calculate the start point (center of the block)
        start_point = (col + block_size // 2, row + block_size // 2)
        
        # Calculate the end point based on the motion vector
        end_point = (start_point[0] + dx, start_point[1] + dy)
        
        # Draw an arrow from the start point to the end point
        cv2.arrowedLine(vis_frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.3)
    
    return vis_frame



