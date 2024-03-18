import numpy as np
from scipy.ndimage import binary_dilation


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

    fill_in_zeros(new_frame)

    return new_frame

def fill_in_zeros(frame):
    print(np.where(frame == [0, 0, 0]).shape)



    # expands the zero regions by adding neighboring pixels until they meet non-zero pixels
    # expanded_zero_regions = binary_dilation(zero_indices, structure=np.ones((3, 3)))


def find_non_zero_neighbors(frame, y, x):
    """
    Expands kernel around center pixel until 4 corners are non-zero

    Args:
        frame: input frame
        y: y-position of center pixel (row)
        x: x-position of center pixel (column)
    
    Returns:
        positions of top_left, top_right, bottom_left, bottom_right relative to center pixel
    """
    pass

def bilinear_interpolate(y, x, top_left, top_right, bottom_left, bottom_right):
    """
    Performs bilinear interpolation for center pixel (x, y)

    Args:
        y: y-position of center pixel (row)
        x: x-position of center pixel (col)
        top_left: top left non-zero neighbor
        top_right: top right non-zero neighbor
        bottom_left: bottom left non-zero neighbor
        bottom_right: bottom right non-zero neighbor

    Returns:
        new pixel value for (x, y)
    """

    # caclulate distances to center pixel
    top_left_dist = np.sqrt((y - top_left[0])**2 + (x - top_left[1])**2)
    top_right_dist = np.sqrt((y - top_right[0])**2 + (x - top_right[1])**2)
    bottom_left_dist = np.sqrt((y - bottom_left[0])**2 + (x - bottom_left[1])**2)
    bottom_right_dist = np.sqrt((y - bottom_right[0])**2 + (x - bottom_right[1])**2)

    total_dist = top_left_dist + top_right_dist + bottom_left_dist + bottom_right_dist

    # x and y are the relative positions of the zero pixel within the square formed by its four neighbors
    


    



