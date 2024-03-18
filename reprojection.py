import numpy as np

def reproject(last_frame, motion_vectors, block_size):
    """
    Converts a full-color image into an image with only the image edges

    Args:
        last_frame: most recent rendered frame
        motion_vectors: matrix of motion vectors corresponding to motion from 2 frames before to the last frame

    Returns:
        A new frame that is a reprojection of the last frame
    """

    # Initialize the new frame with zeros (black image)
    new_frame = np.zeros_like(last_frame)

    height, width = last_frame.shape[:2]

    for (block_y, block_x), (dy, dx) in motion_vectors:
    # Calculate new block position
        new_block_y = block_y + dy
        new_block_x = block_x + dx

        # Ensure the entire block stays within frame boundaries
        if 0 <= new_block_x < width - block_size and 0 <= new_block_y < height - block_size:
            # Copy block from last frame to new position in new frame
            new_block = last_frame[block_y:block_y + block_size, block_x:block_x + block_size]
            new_frame[new_block_y:new_block_y + block_size, new_block_x:new_block_x + block_size] = new_block

            # if np.any(new_block == [0, 0, 0]):
            #     print(f"Block at ({new_block_y}, {new_block_x}):")
            #     print(new_block)

    return new_frame