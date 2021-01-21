# ip

## Program 1: A PROGRAM TO DISPLAY GRAYSCALE IMAGE USING READ AND WRITE OPERATION.
## Description
## Grayscaling 
is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and complete white.To display an image on the window, we have a function cv2.imshow(). This function creates a window and displays the image with the original size.The function has two parameters. The first parameter is a string that describes the name of the window. The second parameter is the image array which contains the image data.cv2.waitKey() function waits for a keyboard interrupt for a specific amount of time in milliseconds. The next function cv2.destroyAllWindows() closes all the windows in which images are displayed.

## Program code:
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

## Output:
![image](https://user-images.githubusercontent.com/72516233/104425714-d9c31300-5535-11eb-8d11-9ed6b741e5cd.png)


## Program 2:DEVELOP A PROGRAM TO PERFORM LINEAR TRANSFORMATION ON IMAGE.

## Description:
A linear transformation is a function from one vector space to another that respects the underlying (linear) structure of each vector space. A linear transformation is also known as a linear operator or map. The range of the transformation may be the same as the domain, and when that happens, the transformation is known as an endomorphism or, if invertible, an automorphism. The two vector spaces must have the same underlying field.Linear transformations are useful because they preserve the structure of a vector space. So, many qualitative assessments of a vector space that is the domain of a linear transformation may, under certain conditions, automatically hold in the image of the linear transformation. For instance, the structure immediately gives that the kernel and image are both subspaces (not just subsets) of the range of the linear transformation.

## Program code
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

## Output:
![image](https://user-images.githubusercontent.com/72516233/104425520-a2546680-5535-11eb-9292-1eb33d037171.png)

## Program code 2.b:
import cv2 
import numpy as np 
  
FILE_NAME = ‘brush.jpg'
img = cv2.imread(FILE_NAME) 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result.jpg', res) 
cv2.waitKey(0)

## Output:
![image](https://user-images.githubusercontent.com/72516233/104426247-8dc49e00-5536-11eb-8d95-507a4966eecf.png)


## Program 3:DEVELOP A PROGRAM TO FIND SUM AND MEAN OF A SET OF IMAGES.
CREATE N NUMBER OF IMAGES AND READ THE DIRECTORY AND PERFORM OPERATION.


## Description:
listdir() returns a list containing the names of the entries in the directory given by path. The list is in arbitrary order. It does not include the special entries '.' and '..' even if they are present in the directory.
append:this method adds a single item to the existing list.

## Program code

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

## Output:
![image](https://user-images.githubusercontent.com/72516233/104429155-2577bb80-553a-11eb-92bd-b90464f10e0c.png)


## Program 4:WRITE A PROGRAM TO CONVERT COLOR IMAGE INTO GRAY SCALE AND BINARY IMAGE.


## Description:
A grayscale (or graylevel) image is simply one in which the only colors are shades of gray. The reason for differentiating such images from any other sort of color image is that less information needs to be provided for each pixel. Gray Scale Image :
Grayscale image contains only single channel. Pixel intensities in this color space is represented by values ranging from 0 to 255. Thus, number of possibilities for one color represented by a pixel is 256.

BW = im2bw(I,level) converts the grayscale image I to binary image BW, by replacing all pixels in the input image with luminance greater than level with the value 1 (white) and replacing all other pixels with the value 0 (black).

This range is relative to the signal levels possible for the image's class. Therefore, a level value of 0.5 corresponds to an intensity value halfway between the minimum and maximum value of the class
## Program code
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

## Output:
![image](https://user-images.githubusercontent.com/72516233/104426936-7639e500-5537-11eb-991e-9b0676cb7d74.png)


## Program 5:WRITE A PROGRAM TO CONVERT COLOR IMAGE INTO DIFFERENT COLOR SPACE.
Description:
## HSV color space :
H : Hue represents dominant wavelength.
S : Saturation represents shades of color.
V : Value represents Intensity.

## LAB color space :
L – Represents Lightness.
A – Color component ranging from Green to Magenta.
B – Color component ranging from Blue to Yellow.
## YUV:
Y refers to the luminance or intensity, and U/V channels represent color information. Y channel is the same as the grayscale image. It represents the intensity values. The U and V channels represent the color information
## HSL (Hue Saturation Lightness): 
very similar to HSV and used in image editing softwares as well
## Grayscale:
image contains only single channel. Pixel intensities in this color space is represented by values ranging from 0 to 255. Thus, number of possibilities for one color represented by a pixel is 256.

## Program code
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


## Output:
![image](https://user-images.githubusercontent.com/72516233/104427227-da5ca900-5537-11eb-9e91-01614764ab98.png)


## Program 6:A PROGRAM TO CREATE AN IMAGE FROM 2D ARRAY.

## Description:
## 2d ARRAY:
Two dimensional array is an array within an array. It is an array of arrays. In this type of array the position of an data element is referred by two indices instead of one. So it represents a table with rows an dcolumns of data. In the below example of a two dimensional array, observer that each array element itself is also an array.
## PROGRAM CODE
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


## Output:
![image](https://user-images.githubusercontent.com/72516233/104428110-e7c66300-5538-11eb-8374-1e9d2100f53a.png)

## 7. PROGRAM TO FIND THE SUM OF NEIGHBORHOOD VALUES OF THE MATRIX.
## Description
Here for each elements of the matrix.We are going to find the corresponding sum of matrix.
numpy.zeros()or np.zeros Python function is used to create a matrix full of zeros.numpy.zeros() in python can be used when you initialize the weights.
## Program code:

import numpy as np
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)

## Output:
Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]
 
 

## 8.C++ Programs to perform operator overloading.
## Program code:
i#include <iostream>
using namespace std;
class matrix
{
 int r1, c1, i, j, a1;
 int a[10][10];

public:int get()
 {
  cout << "Enter the row and column size for the  matrix\n";
  cin >> r1;
  cin >> c1;
   cout << "Enter the elements of the matrix\n";
  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    cin>>a[i][j];

   }
  }
 
 
 };
 void operator+(matrix a1)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] + a1.a[i][j];
    }
   
  }
  cout<<"addition is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }

 };

  void operator-(matrix a2)
 {
 int c[i][j];
  
   for (i = 0; i < r1; i++)
   {
    for (j = 0; j < c1; j++)
    {
     c[i][j] = a[i][j] - a2.a[i][j];
    }
   
  }
  cout<<"subtraction is\n";
  for(i=0;i<r1;i++)
  {
   cout<<" ";
   for (j = 0; j < c1; j++)
   {
    cout<<c[i][j]<<"\t";
   }
   cout<<"\n";
  }
 };

 void operator*(matrix a3)
 {
  int c[i][j];

  for (i = 0; i < r1; i++)
  {
   for (j = 0; j < c1; j++)
   {
    c[i][j] =0;
    for (int k = 0; k < r1; k++)
    {
     c[i][j] += a[i][k] * (a3.a[k][j]);
    }
  }
  }
  cout << "multiplication is\n";
  for (i = 0; i < r1; i++)
  {
   cout << " ";
   for (j = 0; j < c1; j++)
   {
    cout << c[i][j] << "\t";
   }
   cout << "\n";
  }
 };

};

int main()
{
 matrix p,q;
 p.get();
 q.get();
 p + q;
 p - q;
 p * q;
return 0;
}



## Output:
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
6
7
5
8
Enter the row and column size for the  matrix
2
2
Enter the elements of the matrix
2
3
1
4
addition is
 8      10
 6      12
subtraction is
 4      4
 4      4
multiplication is
 19     46
 18     47


## 9. Finding the negative
## Descripion:
Image is also known as a set of pixels. When we store an image in computers or digitally, it’s corresponding pixel values are stored. So, when we read an image to a variable using OpenCV in Python, the variable stores the pixel values of the image. When we try to negatively transform an image, the brightest areas are transformed into the darkest and the darkest areas are transformed into the brightest.
## Program code:
from pylab import*
from skimage import img_as_float
skI = imread("pic.jpg");    
subplot(1, 2, 1),
 
#displaying the RGB image
imshow(skI);
title("Original image");
 
#levels of the 8-bit image
L = 2 ^ 8;    
                     
neg = (L - 1) - skI;
subplot(1, 2, 2),
 
#displaying the negative image
imshow(neg);
title("Negative Image") 

## Output:
![image](https://user-images.githubusercontent.com/72516233/105158865-4c4a6a80-5ac3-11eb-9a53-29d968fda715.png)

## 
## 10.Threshold
## Program code:
import cv2  
import numpy as np  
 

image1 = cv2.imread('fan.jpg')  
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
                                                                                                                                                                                                                                                                                                   
cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)
     
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()

## Output:
![image](https://user-images.githubusercontent.com/72516233/105330002-a6603400-5b86-11eb-8acf-919d3fa9109f.png)

## 11.Histogram
## Program code:
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('quote.jpg',0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show() 

## Output:
![image](https://user-images.githubusercontent.com/72516233/105336769-77e65700-5b8e-11eb-887c-45f5fd99905a.png)



## 12.Program to perform Gamma transformation.
## Program code:

import cv2
import numpy as np
img=cv2.imread("quote1.jpg")
gamma_two_point_two=np.array(255*(img/255)**2.2,dtype='uint8')
gamma_point_four=np.array(255*(img/255)**0.24,dtype='uint8')
img3=cv2.hconcat([gamma_two_point_two,gamma_point_four])
cv2.imshow('quote1',img3)
cv2.waitKey(0)

## Output:
![image](https://user-images.githubusercontent.com/72516233/105338262-36ef4200-5b90-11eb-9249-3a4ee6df13ed.png)




