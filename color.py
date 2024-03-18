import numpy as np
import cv2

color = np.array([109, 114,   0], dtype=np.uint8)  # Convert to uint8 for OpenCV compatibility
color_image = np.full((100, 100, 3), color, dtype=np.uint8)  # Create a 100x100 image filled with the color

cv2.imshow('Color', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()