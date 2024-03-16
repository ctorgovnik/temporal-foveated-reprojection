import cv2
import numpy as np

video = 'building_sample_video.mp4'

cap = cv2.VideoCapture(video)

# Read the first frame
ret, prev_frame = cap.read()
# ret2, curr_frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("height: ", frame_height)
print("width: ",frame_width)
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(fps)
new_fps = 10

out = cv2.VideoWriter('building_motion_vectors.mp4', cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (frame_width, frame_height))
# Convert the first two frames to grayscale

# frame_index = 0
# prev_frame = None
# prev_gray = None
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

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)


    motion_vectors = estimate_motion_vectors(prev_gray, curr_gray, 16, 7)

    visualized_frame = visualize_motion_vectors(curr_frame, motion_vectors)
    
    # Write or display the visualized frame
    out.write(visualized_frame)


    prev_gray = curr_gray
cap.release()
out.release()
cv2.destroyAllWindows()

motion_vectors = estimate_motion_vectors(prev_gray, curr_gray, 16, 7)

print(motion_vectors)


# Assuming `motion_vectors` are calculated from `estimate_motion_vectors()`
visualized_frame = visualize_motion_vectors(curr_gray, motion_vectors)

# Display the frame with motion vectors
cv2.imshow("Motion Vectors", visualized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the visualized frame to a file
cv2.imwrite('motion_vectors_visualization.jpg', visualized_frame)
