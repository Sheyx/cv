#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:39:00 2020

@author: sheyx
"""

import os

import cv2
import numpy as np
from PIL import Image

# Path for face image database
path = 'facesLib'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = []
    for f in os.listdir(path):
        if f.split('.')[1] == 'jpg':
            imagePaths.append(path + '/' + f)
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, np.uint8)
        id = int(os.path.split(imagePath)[-1].split("-")[0])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('training/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
