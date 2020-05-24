#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:57:50 2020

@author: sheyx

This is script for tracking ball
"""

import math
import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 600)  # set Width
cap.set(4, 600)  # set Height
time.sleep(0.1)


def createPath(img):
    h, w = img.shape[:2]
    return np.zeros((h, w, 3), np.uint8)


hsv_min = np.array((29, 54, 148), np.uint8)
hsv_max = np.array((44, 216, 255), np.uint8)
t = 0

# Начальные координаты трекинга
lastx = 0
lasty = 0

ret, img = cap.read()
path = createPath(img)

while True:
    ret, img = cap.read()
    # Инициализация фильтров
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Фильтрация по диапазону цветов
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    # Морфологическое преобразование - удаление шумов
    st1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21), (10, 10))
    st2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11), (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, st1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, st2)
    # Сглаживание
    thresh = cv2.GaussianBlur(thresh, (5, 5), 2)
    # Определение положения
    x, y, w, h = cv2.boundingRect(thresh)
    # Определение минимального размера объекта
    if w > 10 and h > 10:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        nx = int(x + w / 2)
        ny = int(y + h / 2)
        if lastx > 0 and lasty > 0:
            ln = math.sqrt(math.pow(nx - lastx, 2) + math.pow(ny - lasty, 2))
            if ln > 50:
                cv2.line(path, (lastx, lasty), (nx, ny), (40, 115, 10), 2)
                lastx = nx
                lasty = ny
        else:
            lastx = nx
            lasty = ny
    # FPS
    dt = time.time() - t
    t = time.time()
    text = '%0.1f' % (1. / dt)
    cv2.putText(img, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)

    # Накладываем
    img = cv2.add(img, path)

    # Вывод изображения
    cv2.imshow('result', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
