import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from plaka_konum_alg import plaka_konum
from karakter_okuma_alg import plakaTani


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


target_fps = 10
wait_time = int(1000 / target_fps)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    ###### Our operations on the frame come here
    
    plaka = plaka_konum(frame)                               #   plakanın konumunu aldık
    plakaImg,plakaKarakter = plakaTani(frame, plaka)         #   konumdaki karakterleri okuyoruz
    
    ##### Display the resulting frame
    cv2.imshow('frame', plakaImg)
    if cv2.waitKey(2)  == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()