import requests
import cv2
import numpy as np
cap = 'http://192.168.0.16:8080/shot.jpg'
while True:
    img_resp = requests.get(cap)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    cv2.imshow('frame', img)
    key = cv2.waitKey(30)
    if key == 27:
        break