import numpy
from numpy import asarray
from PIL import Image
from flask import Flask, request, render_template
from pathlib import Path
from datetime import datetime
from lbphAlgorithm import getFeatureVector
from face_detection import face_detection
import pandas as pd
import cv2 as cv
import math

app = Flask(__name__)

def readImageFeature(queryEmbedding):
  df = pd.read_csv('data/csvfiles/data_3.csv', header=None)
  # lấy column đầu tiên 
  first_column = df.columns[0]
  last_column = df.columns[257]
  #print(last_column)
  # xóa column đầu tiên
  df = df.drop([first_column], axis=1)
  df = df.drop([last_column], axis=1)
  #print(df)
  df = df.iloc[1:]
  # print(df)
  arr = df.to_numpy()
  print("\nArray vector feature\n")
  print(arr)
  # xóa row đầu tiên
  #df = df.drop(index=df.index[0], axis=0, inplace=True)
  minn = math.inf
  res = list()
  #print(arr)
  index = 0
  i = 0
  for x in arr:
    # print(x)
    # print(asarray(queryEmbedding))
    dist = numpy.linalg.norm(x - asarray(queryEmbedding))
    res.append(abs(dist))
    if minn > abs(dist):
      minn = abs(dist)
      index = i
    i+=1
  #print(df)
  #print(minn)
  #print(index)
  return index



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        df = pd.read_csv('data/csvfiles/data_3.csv', header=None)
        first_column = df.columns[0]
        df = df.drop([first_column], axis=1)
        df = df.iloc[1:]
        last_column = df.iloc[: , -1]
        # từ pandas -> thành array python
        # transfer cot cuoi thanh list
        last_column = last_column.tolist()
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        print("Namdd")
        detectionQueryImage = face_detection(uploaded_img_path)
        tmp_image = cv.cvtColor(detectionQueryImage, cv.COLOR_BGR2RGB)
        # Run search
        query = getFeatureVector(tmp_image)
        print("\nQuery vector\n")
        print(query)
        print(last_column)
        resIndex = readImageFeature(query)
        print("\nResult path\n")
        print(last_column[resIndex])


        return render_template('index.html',
                               query_path=uploaded_img_path,
                               score=last_column[resIndex])
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
