#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:30:20 2020

@author: sheyx
"""
import cv2

faceCascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
cap.set(3, 800)  # set Width
cap.set(4, 600)  # set Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
