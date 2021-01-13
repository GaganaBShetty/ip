# ip

program 1:a program to display grayscale image.
description:Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.

import cv2
import numpy as np

image = cv2.imread('artgirl.jpg')
image = cv2.resize(image, (0, 0), None, .99, .99)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('Art_Girl', numpy_horizontal_concat)
cv2.waitKey()

![image](https://user-images.githubusercontent.com/72516233/104419669-cac47b00-559e-11eb-9866-9b91ebf0fe84.png)
