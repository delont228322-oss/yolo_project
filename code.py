import cv2
import numpy as np
import pygame
import random as rd
import os
from ultralytics import YOLO

pygame.mixer.init()
if os.path.exists("jingle_bells.mp3"):
    pygame.mixer.music.load("jingle_bells.mp3")

model = YOLO("yolov8n.pt")

def make_transparent(img):
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    rgba[:, :, 3] = mask
    return rgba

hat_img = make_transparent(cv2.imread("hat.png"))
snowflakes = [[rd.randint(0, 640), rd.randint(0, 480)] for _ in range(60)]
HAT_SIZE, HAT_Y = 0.6, 0.2

def overlay_hat(frame, hat, x, y, w, h):
    try:
        hw, hh = int(w * HAT_SIZE), int(w * HAT_SIZE * 0.8)
        hat_res = cv2.resize(hat, (hw, hh))
        nx, ny = x + (w - hw) // 2, y - int(hh * HAT_Y)
        y1, y2 = max(0, ny), min(frame.shape[0], ny + hh)
        x1, x2 = max(0, nx), min(frame.shape[1], nx + hw)
        h_part = hat_res[max(0, -ny):max(0, -ny)+(y2-y1), max(0, -nx):max(0, -nx)+(x2-x1)]
        alpha = h_part[:, :, 3] / 255.0
        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha * h_part[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c])
    except: pass
    return frame

cap, music_playing = cv2.VideoCapture(0), False
