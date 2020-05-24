#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:30:20 2020

@author: sheyx
"""
import cv2
import time

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Подцепляем камеру
cap = cv2.VideoCapture(1)
#cap.set(3, 640)  # set Width
#cap.set(4, 480)  # set Height
ret, img = cap.read()
time.sleep(1)

faceID = int(input('Enter ID = '))
count = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        count += 1

    #SaveFace
        cv2.imwrite('facesLib/' + str(faceID) + '-img-' + str(count) + '.jpg', roi_gray)
    #Show window
    cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
