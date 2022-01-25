import cv2 as cv
import numpy as np

face_cascade =  cv.CascadeClassifier('haar_face.xml')
mouth_cascade = cv.CascadeClassifier('haar_mouth.xml')


cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv.flip(img,1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    thresh, bw_image = cv.threshold(gray, 80, 255,cv.THRESH_BINARY) #bw means black and white
    #detectMultiScale(image, scalefactor,minNeighbors)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_inbw = face_cascade.detectMultiScale(bw_image, 1.1, 4)

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)

        mouth_roi = mouth_cascade.detectMultiScale(gray, 1.5, 4)  #roi is region of interest

        if len(mouth_roi) == 0:
            cv.putText(img, "Mask Weared", (x,y), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), thickness=2)
        else:
            for (tx,ty,tw,th) in mouth_roi:
                if (y < ty < y + h):
                        cv.putText(img,"mask not weared", (20,20), cv.FONT_HERSHEY_TRIPLEX, 1.0,
                                    (0,0,255), thickness=2)
                        break
    cv.imshow('Mask Detection',img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
