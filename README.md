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

Description:A linear transformation is a function from one vector space to another that respects the underlying (linear) structure of each vector space. A linear transformation is also known as a linear operator or map. The range of the transformation may be the same as the domain, and when that happens, the transformation is known as an endomorphism or, if invertible, an automorphism. The two vector spaces must have the same underlying field.Linear transformations are useful because they preserve the structure of a vector space. So, many qualitative assessments of a vector space that is the domain of a linear transformation may, under certain conditions, automatically hold in the image of the linear transformation. For instance, the structure immediately gives that the kernel and image are both subspaces (not just subsets) of the range of the linear transformation.

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
path = 'C:\Users\MCA\Downloads\gagana'
imgs = []

files = os.listdir(path)
for file in files:
    filepath=path+"\\"+file
    imgs.append(cv2.imread(filepath))
i=0
im = []
for im in imgs:
    #cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1
cv2.imshow("sum of four pictures",im)
meanImg = im/len(files)
cv2.imshow("mean of four pictures",meanImg)
cv2.waitKey(0)

Output:
![image](https://user-images.githubusercontent.com/72516233/104429155-2577bb80-553a-11eb-92bd-b90464f10e0c.png)


Program 4:Write a program to convert color image into gray scale and binary image.

Description:A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. Gray Scale Image :
Grayscale image contains only single channel. Pixel intensities in this color space is represented by values ranging from 0 to 255. Thus, number of possibilities for one color represented by a pixel is 256.

BW = im2bw(I,level) converts the grayscale image I to binary image BW, by replacing all pixels in the input image with luminance greater than level with the value 1 (white) and replacing all other pixels with the value 0 (black).

This range is relative to the signal levels possible for the image's class. Therefore, a level value of 0.5 corresponds to an intensity value halfway between the minimum and maximum value of the class

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

Description:
HSV color space :
H : Hue represents dominant wavelength.
S : Saturation represents shades of color.
V : Value represents Intensity.

LAB color space :
L – Represents Lightness.
A – Color component ranging from Green to Magenta.
B – Color component ranging from Blue to Yellow.

YUV:Y refers to the luminance or intensity, and U/V channels represent color information. 





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
![image](https://user-images.githubusercontent.com/72516233/104427227-da5ca900-5537-11eb-9e91-01614764ab98.png)


Program 6:Develop a program to create an image from 2D array.

import numpy as np
from PIL import Image
import cv2 as c 
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255]   #Blue right side

img = Image.fromarray(array)
img.save('COLORS.jpg')
img.show()
c.waitKey(0)


Output:
![image](https://user-images.githubusercontent.com/72516233/104428110-e7c66300-5538-11eb-8374-1e9d2100f53a.png)
