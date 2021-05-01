
import numpy as np
import matplotlib.pyplot as plt      # importing the matplotlib and cv2 for plottig and reading the image file
import cv2
import pandas as pd
# from google.colab import files       # uploading the file to google colabs from PC

# uploaded = files.upload()

# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))
# from google.colab.patches import cv2_imshow        # converting the image to gray scale image    
img= cv2.imread("data/test/harry_test_2.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Scale', gray_img)
def Binarypattern(im):                               # creating function to get local binary pattern
    img= np.zeros_like(im)
    n=3                                              # taking kernel of size 3*3
    for i in range(0,im.shape[0]-n):                 # for image height
      for j in range(0,im.shape[1]-n):               # for image width
            x  = im[i:i+n,j:j+n]                     # reading the entire image in 3*3 format
            center       = x[1,1]                    # taking the center value for 3*3 kernel
            img1        = (x >= center)*1.0          # checking if neighbouring values of center value is greater or less than center value
            img1_vector = img1.T.flatten()           # getting the image pixel values 
            img1_vector = np.delete(img1_vector,4)  
            digit = np.where(img1_vector)[0]         
            if len(digit) >= 1:                     # converting the neighbouring pixels according to center pixel value
                num = np.sum(2**digit)              # if n> center assign 1 and if n<center assign 0
            else:                                    # if 1 then multiply by 2^digit and if 0 then making value 0 and aggregating all the values of kernel to get new center value
                num = 0
            img[i+1,j+1] = num
    return(img)

imgLBP=Binarypattern(gray_img)
cv2.imshow('Gray Scale 2', imgLBP)
             # calling the LBP function using gray image
# chuyển ảnh từ ma trận -> vector
vectorLBP = imgLBP.flatten()               # for histogram using the vector form of image pixels
#pd.DataFrame(vectorLBP).to_csv("data/csvfiles/harry_.csv")
print("Vector LBP")
print(vectorLBP)
fig=plt.figure(figsize=(20,8))             #subplotting the gray, LBP and histogram 
ax  = fig.add_subplot(1,3,1)
ax.imshow(gray_img)
ax.set_title("gray scale image")
ax  = fig.add_subplot(1,3,2)
ax.imshow(imgLBP,cmap="gray")
ax.set_title("LBP converted image")
ax  = fig.add_subplot(1,3,3)
space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228,
229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
# space giúp chia khoảng giá trị trên Histogram
print('With manual value')
print(ax.hist(vectorLBP,bins=space)) # in ra giá trị của histogram
print('With bins value')
print(ax.hist(vectorLBP,bins=2**8)) # in ra giá trị của histogram
freq,lbp, tmp = ax.hist(vectorLBP,bins=space, edgecolor='black')
ax.set_ylim(0,80000)
lbp = lbp[:-1]
print('lbp \n')
print(lbp)
print('frequencies \n')
print(freq)
print('tmp \n')
print(tmp)
## print the LBP values when frequencies are high
largeTF = freq > 5000
for x, fr in zip(lbp[largeTF],freq[largeTF]):
     ax.text(x,fr, "{:6.0f}".format(x),color="magenta")
ax.set_title("LBP histogram")
plt.show()