#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:57:50 2020

@author: sheyx

This is script for tracking ball
"""


import cv2
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3,800) # set Width
cap.set(4,600) # set Height

hsv_min = np.array((28, 43, 136), np.uint8)
hsv_max = np.array((49, 221, 255), np.uint8)


while True:
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imshow('video',thresh)
    

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()