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
cap.set(3,600) # set Width
cap.set(4,600) # set Height

hsv_min = np.array((29, 54, 148), np.uint8)
hsv_max = np.array((44, 216, 255), np.uint8)


while True:
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Фильтрация по диапазону цветов
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    #Морфологическое преобразование - удаление шумов
    st1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21), (10, 10))
    st2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11), (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, st1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, st2)
    #Сглаживание
    thresh = cv2.GaussianBlur(thresh, (5, 5), 2)
    #Определение положения 
    x,y,w,h = cv2.boundingRect(thresh)
    #Определение минимального размера объекта
    if w > 10 and h > 10:
        cv2.rectangle(img,(x, y),(x+w, y+h),(0,0,255),3) 


    cv2.imshow('result', img) 
    
    

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()