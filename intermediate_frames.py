import cv2
import numpy as np

video = 'building_sample_video.mp4'

cap = cv2.VideoCapture(video)

# Read the first frame
ret, prev_frame = cap.read()
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("height: ", frame_height)
print("width: ",frame_width)
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(fps)
new_fps = 10

out = cv2.VideoWriter('building_output_new10fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (frame_width, frame_height))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# Create a black frame
remap = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)



frame_index = 0
prev_frame = None
prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)







    
    # if frame_index % 2 == 0:
    #     # For even indices, just update the variables
    #     # This is the first frame of the pair, which we keep as is
    #     if frame_index > 0:  # Skip writing for the very first frame to adjust the sequence
    #         out.write(prev_frame)  # Write the previous real frame
    # else:
    #     # For odd indices, generate and write an artificial frame based on optical flow
    #     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #     h, w = gray.shape[:2]
    #     x, y = np.meshgrid(np.arange(w), np.arange(h))
    #     flow_map_x, flow_map_y = x + flow[..., 0], y + flow[..., 1]
        
    #     flow_map_x = flow_map_x.astype(np.float32)
    #     flow_map_y = flow_map_y.astype(np.float32)

    #     # Apply the flow map to each color channel of the previous frame
    #     remapped_frames = []
    #     for channel in cv2.split(prev_frame):
    #         remapped_channel = cv2.remap(channel, flow_map_x, flow_map_y, cv2.INTER_LINEAR)
    #         remapped_frames.append(remapped_channel)
    #     remap = cv2.merge(remapped_frames)
        
        # out.write(remap)  # Write the artificially generated frame

    # Update variables for the next iteration
    # prev_frame = frame
    # prev_gray = gray
    # frame_index += 1

# After the loop, write the last real frame if the total number of frames is odd
# if frame_index % 2 == 0:
#     out.write(prev_frame)

cap.release()
out.release()

