import numpy as np
import cv2
import keyboard
import time


palm_cascade = cv2.CascadeClassifier("palm1.xml")
fist_cascade = cv2.CascadeClassifier("fist.xml")

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    palms = palm_cascade.detectMultiScale(gray, 1.3, 5)
    fists = fist_cascade.detectMultiScale(gray, 1.3, 5)   
    for (x,y,w,h) in palms:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        keyboard.press('down')
        time.sleep(0.01)
        if(fist_cascade.detectMultiScale(gray, 1.4, 5)):
            break


    for (x,y,w,h) in fists:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        keyboard.press('up')
        time.sleep(0.01)
        if(palm_cascade.detectMultiScale(gray, 1.4, 5)):
            break
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()