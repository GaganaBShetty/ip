# ip

Program 1:a program to display grayscale image using read and write operation .

description:Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.To display an image on the window, we have a function cv2.imshow(). This function creates a window and displays the image with the original size.The function has two parameters. The first parameter is a string that describes the name of the window. The second parameter is the image array which contains the image data.cv2.waitKey() function waits for a keyboard interrupt for a specific amount of time in milliseconds. The next function cv2.destroyAllWindows() closes all the windows in which images are displayed.

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

Output:
![image](https://user-images.githubusercontent.com/72516233/104425714-d9c31300-5535-11eb-8d11-9ed6b741e5cd.png)


Program 2:Develop a program to perform linear transformation on image.

import cv2
import numpy as np
FILE_NAME = 'art.jpg'
try: 
    img = cv2.imread(FILE_NAME) 
    (height, width) = img.shape[:2] 
    res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC) 
    cv2.imwrite('result.jpg', res) 
    cv2.imshow('image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0)
except IOError: 
    print ('Error while reading files !!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)

Output:
![image](https://user-images.githubusercontent.com/72516233/104425520-a2546680-5535-11eb-9292-1eb33d037171.png)

Program 2.b:
import cv2 
import numpy as np 
  
FILE_NAME = ‘brush.jpg'
img = cv2.imread(FILE_NAME) 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result.jpg', res) 
cv2.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72516233/104426247-8dc49e00-5536-11eb-8d95-507a4966eecf.png)


Program 3:Develop a program to find sum and mean of a set of images.
Create n number of images and read the directory and perform operation.
import cv2
import os
path = "E:\ART"
imgs=[]
dirs=os.listdir(path)

for file in dirs:
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
    
i=0
for im in imgs:
    cv2.imshow(dirs[i],imgs[i])
    i=i+1
print(i)
cv2.imshow('sum',len(im))
cv2.imshow('mean',len(im)/im)
cv2.waitKey(0)


Output:
![image](https://user-images.githubusercontent.com/72516233/104426602-0166ab00-5537-11eb-852b-e1f13fac9f83.png)


Program 4:Write a program to convert color image into gray scale and binary image.


import cv2
img = cv2.imread("BRUSH.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Binary Image",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

Output:
![image](https://user-images.githubusercontent.com/72516233/104426936-7639e500-5537-11eb-991e-9b0676cb7d74.png)


Program 5:Write a program to convert color image into different color space.
import cv2
img = cv2.imread("ACRYLIC.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

cv2.imshow("GRAY image",gray)
cv2.waitKey(0)

cv2.imshow("HSV image",hsv)
cv2.waitKey(0)

cv2.imshow("LAB image",lab)
cv2.waitKey(0)

cv2.imshow("HLS image",hls)
cv2.waitKey(0)

cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.destroyAllWindows()


Output:
