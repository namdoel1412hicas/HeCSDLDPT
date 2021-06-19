
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

def Binarypattern(im):                               # creating function to get local binary pattern
    # Trả về mảng các giá trị 0 với cùng dạng và loại với array đưa vào
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



def getFeatureVector(img):
  # img= cv2.imread(filePath)
  gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  cv2.imshow('Gray Scale', gray_img)
  imgLBP=Binarypattern(gray_img)
  cv2.imshow('Gray Scale 2', imgLBP)
              # calling the LBP function using gray image
  # chuyển ảnh từ ma trận -> vector
  vectorLBP = imgLBP.flatten()               # for histogram using the vector form of image pixels
  #pd.DataFrame(vectorLBP).to_csv("data/csvfiles/harry_.csv")
  # print("Vector LBP")
  # print(vectorLBP)
  fig=plt.figure(figsize=(20,8))             #subplotting the gray, LBP and histogram 
  ax  = fig.add_subplot(1,3,1)
  # tmp_image = cv2.cvtColor(gray_img,cv2.COLOR_BGR2GRAY)
  # dùng cmap='gray'
  ax.imshow(gray_img,cmap="gray")
  ax.set_title("gray scale image")
  ax  = fig.add_subplot(1,3,2)
  ax.imshow(imgLBP,cmap="gray")
  ax.set_title("LBP converted image")
  ax  = fig.add_subplot(1,3,3)
  space = list()
  for i in range(0, 257):
    space.append(i)

  print(space)
  # space giúp chia khoảng giá trị trên Histogram
  # ========
  # print('With manual value')
  # print(ax.hist(vectorLBP,bins=space)) # in ra giá trị của histogram
  # print('With bins value')
  # print(ax.hist(vectorLBP,bins=2**8)) # in ra giá trị của histogram
  freq, lbp, tmp = ax.hist(vectorLBP,bins=space, edgecolor='black')
  # freq là feature vector gồm 256 giá trị cần phải lưu
  ax.set_ylim(0,8000)
  lbp = lbp[:-1]
  # print('lbp \n')
  # print(lbp)
  # print('frequencies \n')
  # print(freq)
  # print('tmp \n')
  # print(tmp)
  ## print the LBP values when frequencies are high
  largeTF = freq > 5000
  for x, fr in zip(lbp[largeTF],freq[largeTF]):
      ax.text(x,fr, "{:6.0f}".format(x),color="magenta")
  ax.set_title("LBP histogram")
  #plt.show()
  # trả về 
  return freq
  
# getFeatureVector("data/test/harry_test_1.jfif")