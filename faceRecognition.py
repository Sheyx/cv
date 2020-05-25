#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:30:20 2020

@author: sheyx
"""
import cv2
import time
import numpy as np

faceCascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

#Подцепляем камеру
cap = cv2.VideoCapture(0)
#cap.set(3, 640)  # set Width
#cap.set(4, 480)  # set Height


#FPS timer
t = 0

names = ['None', 'Dmitry', 'Stefa']
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('faceRec/training/trainer.yml')
cascadePath = "faceRec/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


id = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 50):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)



        #FPS
    dt = time.time() - t
    t = time.time()
    text = '%0.1f' % (1. / dt)
    cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)

    #Show window
    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
