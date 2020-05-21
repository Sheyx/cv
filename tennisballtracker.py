#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:57:50 2020

@author: sheyx

This is script for tracking ball
"""


import cv2

cap = cv2.VideoCapture(0)
cap.set(3,320) # set Width
cap.set(4,240) # set Height

while True:
    ret, img = cap.read()
   

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()