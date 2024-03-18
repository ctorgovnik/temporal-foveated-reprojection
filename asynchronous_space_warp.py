import cv2
import numpy as np
import motion_vector_estimation as mv
import reprojection as rp

video = 'building_sample_video.mp4'



def write_frames(video, number_frames, new_fps, output_name):

    cap = cv2.VideoCapture(video)

    # Read the first frame
    ret, prev_frame = cap.read()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    out = cv2.VideoWriter(f'{output_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (frame_width, frame_height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


    i = 0
    while cap.isOpened():
        if i >= number_frames:
            break

        ret, curr_frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if i>0 and i%2 ==0:

            
            motion_vectors = mv.estimate_motion_vectors(prev_gray, curr_gray, 16, 7)

            reprojected_frame = rp.reproject(curr_frame, motion_vectors, 16)
            reproj = np.array(reprojected_frame)
            # np.savetxt("reprojected_frame.csv", reproj.reshape(-1, reproj.shape[-1]), fmt='%d', delimiter=",")

            prev_gray = reprojected_frame

        # visualized_frame = mv.visualize_motion_vectors(curr_frame, motion_vectors)
        
        # Write or display the visualized frame
        # out.write(visualized_frame)
            out.write(reprojected_frame)
        else:
            out.write(curr_frame)
            prev_gray = curr_gray
        i += 1

        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

cap = cv2.VideoCapture(video)

# Read the first frame
# ret, prev_frame = cap.read()
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# ret, curr_frame = cap.read()
# curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

# motion_vectors = mv.estimate_motion_vectors(prev_gray, curr_gray, 16, 7)
# arr = np.array(motion_vectors)

# print(arr.shape)

    # Assuming `motion_vectors` are calculated from `estimate_motion_vectors()`
    # visualized_frame = mv.visualize_motion_vectors(curr_gray, motion_vectors)

    # Display the frame with motion vectors
    # cv2.imshow("Motion Vectors", visualized_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optionally, save the visualized frame to a file
    # cv2.imwrite('motion_vectors_visualization.jpg', visualized_frame)

    


write_frames(video, 20, 10, "reprojected_video")