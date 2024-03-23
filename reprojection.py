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

    filled_frame = fill_in_zeros(new_frame)

    return filled_frame

def fill_in_zeros(frame):
    """
    Finds nearest non-zero neighbors and performs bilinear interpolation for each zero pixel in frame

    Args:
        frame: frame from video that contains some pixels with value of zero

    Returns:
        frame with all zeros filled in based on nearest neighbors
    """
    # Find indices where all channels are 0 (i.e., black pixels)
    black_pixels = np.where(np.all(frame == [0, 0, 0], axis=-1))
    
    rows, cols = black_pixels
    for y, x in zip(rows, cols):
        top_left, top_right, bottom_left, bottom_right = find_non_zero_neighbors(frame, y, x)
        new_pixel = bilinear_interpolate(frame, y, x, top_left, top_right, bottom_left, bottom_right)
        # print(f'Before: {frame[y, x, :]}')
        frame[y, x, :] = new_pixel
        # print(f'After: {frame[y, x, :]}')
    return frame

    


    




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
    height, width, _ = frame.shape

    top_left = [0, 0]
    top_right = [0,0]
    bottom_right = [0, 0]
    bottom_left = [0, 0]

    while (True):

        if y + top_left[0] > 0 and x + top_left[1] > 0 and np.all(frame[y + top_left[0], x + top_left[1]] == 0):
            top_left[0] -= 1
            top_left[1] -= 1
            continue
        
        if y + top_right[0] > 0 and x + top_right[1] < width - 1 and np.all(frame[y + top_right[0], x + top_right[1]] == 0):
            top_right[0] -= 1
            top_right[1] += 1
            continue

        if y + bottom_left[0] < height - 1 and x + bottom_left[1] > 0 and np.all(frame[y + bottom_left[0], x + bottom_left[1]] == 0):
            bottom_left[0] += 1
            bottom_left[1] -= 1
            continue

        if y + bottom_right[0] < height - 1 and x + bottom_right[1] < width - 1 and np.all(frame[y + bottom_right[0], x + bottom_right[1]] == 0):
            bottom_right[0] += 1
            bottom_right[1] += 1
            continue
    
        break

    

    return top_left, top_right, bottom_left, bottom_right
 

        


    pass

def bilinear_interpolate(frame, y, x, top_left, top_right, bottom_left, bottom_right):
    """
    Performs bilinear interpolation for center pixel (x, y)

    Args:
        frame: current frame
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

    top_left_weight = safe_inverse(top_left_dist)
    top_right_weight = safe_inverse(top_right_dist)
    bottom_left_weight = safe_inverse(bottom_left_dist)
    bottom_right_weight = safe_inverse(bottom_right_dist)

    total_weights = top_left_weight + top_right_weight + bottom_left_weight + bottom_right_weight

    # get pixel values
    top_left_val = frame[y + top_left[0], x + top_left[1], :]
    top_right_val = frame[y + top_right[0], x + top_right[1], :]
    bottom_left_val = frame[y + bottom_left[0], x + bottom_left[1], :]
    bottom_right_val = frame[y + bottom_right[0], x + bottom_right[1], :]
    # print(f'top left value: {top_left_val}')
    # print(f'top right value: {top_right_val}')
    # print(f'bottom left value: {bottom_left_val}')
    # print(f'bottom right value: {bottom_right_val}')




    interpolated_pixel = (top_left_weight * top_left_val + top_right_weight * top_right_val + bottom_left_weight * bottom_left_val + bottom_right_weight * bottom_right_val) /total_weights 
    

    return interpolated_pixel


    # x and y are the relative positions of the zero pixel within the square formed by its four neighbors
    

def safe_inverse(distance, epsilon=1e-10):
    """Returns a safe inverse of distance, avoiding division by zero."""
    return 1 / (distance if distance != 0 else epsilon)
