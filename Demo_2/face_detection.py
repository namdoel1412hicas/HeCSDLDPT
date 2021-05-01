import cv2 as cv
from numpy import asarray
from PIL import Image
from os import listdir
from matplotlib import pyplot
from lbphAlgorithm import getFeatureVector
import pandas as pd
import numpy

def face_detection(filename, required_size=(160, 160)):
  # Read image from your local file system
  original_image = cv.imread(filename)

  # Convert color image to grayscale for Viola-Jones
  grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

  # Load the classifier and create a cascade object for face detection
  face_cascade = cv.CascadeClassifier('data/lib/haarcascade_frontalface_alt.xml')

  detected_faces = face_cascade.detectMultiScale(grayscale_image)
  # .....
  # convrt BGR to RGB
  original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
  pixels = asarray(original_image)

  face = 1
  face_array = 1

  column, row, width, height = detected_faces[0]
  print(column)
  print(row) 
  print(width)
  print(height)
  # disable rectangle
  # cv.rectangle(
  #     original_image,
  #     (column, row),
  #     (column + width, row + height),
  #     (0, 255, 0),
  #     2
  # )
  
  # for (column, row, width, height) in detected_faces:
  #     cv.rectangle(
  #         original_image,
  #         (column, row),
  #         (column + width, row + height),
  #         (0, 255, 0),
  #         2
  #     )
  x1, y1 = abs(column), abs(row)
  x2, y2 = column + width, row + height
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  print(face_array)
  return face_array

def fetListFeatureVectors(folder):
  folder = 'data/Harry/'
  i = 1
  features = list()
  # enumerate files
  for filename in listdir(folder):
    # path
    path = folder + filename
    # get face
    face = face_detection(path)
    print(i, face.shape)
    # plot
    pyplot.subplot(2, 7, i)
    pyplot.axis('off')
    # khi chuyển sang bằng hàm asarray, ảnh sẽ có dạng BGR -> phải chuyển về RGB
    pyplot.imshow(face)
    tmp_image = cv.cvtColor(face, cv.COLOR_BGR2RGB)
    featureVct = getFeatureVector(tmp_image)
    featureVct = numpy.append(featureVct, path)
    features.append(featureVct)
    cv.imshow("harry", tmp_image)
    i += 1
  saveFeatureInCSV(features, 'data/csvfiles/data_2.csv')
  pyplot.show()

def saveFeatureInCSV(data, dataFilePath):
  pd.DataFrame(data).to_csv(dataFilePath)

fetListFeatureVectors("")