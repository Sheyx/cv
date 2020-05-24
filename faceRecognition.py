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
cap = cv2.VideoCapture(1)
#cap.set(3, 640)  # set Width
#cap.set(4, 480)  # set Height
ret, img = cap.read()
time.sleep(1)
h, w = img.shape[:2]
roi_color = np.zeros((h, w, 3), np.uint8)

#FPS timer
t = 0


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

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
