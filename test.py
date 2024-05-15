import cv2
import numpy as np
from augraphy import *

# Load your image using OpenCV
image = cv2.imread('../ImageSteles/00043.jpg')

shadowcast = ShadowCast(shadow_side = "bottom",
                        shadow_vertices_range = (2, 3),
                        shadow_width_range=(0.5, 0.8),
                        shadow_height_range=(0.5, 0.8),
                        shadow_color = (0, 0, 0),
                        shadow_opacity_range=(0.5,0.6),
                        shadow_iterations_range = (1,2),
                        shadow_blur_kernel_range = (101, 301),
                        )

img_shadowcast = shadowcast(image)
cv2.imshow("shadowcast", img_shadowcast)

cv2.waitKey(0)
cv2.destroyAllWindows()