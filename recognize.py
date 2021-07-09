import cv2
import numpy as np
import xlwrite
import time
import sys
from playsound import playsound
start = time.time()
period = 8
casclf = cv2.CascadeClassifier('C:\\Users\\hrtripathi\\Desktop\\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\hrtripathi\\Desktop\\Trainer\\trainer.yml')
flag = 0
id = 0
filename = 'filename'
dict = {
    'item1': 1
}
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = casclf.detectMultiScale(gray, 1.3, 7);
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2);
        id, conf = recognizer.predict(roi_gray)
        if (conf < 85):
            if (id == 1):
                id = 'Harsh'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 1, id, 'yes')
                    dict[str(id)] = str(id)

            elif (id == 2):
                id = 'Hitarth'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 2, id, 'yes')
                    dict[str(id)] = str(id)

            elif (id == 3):
                id = 'Shivam'
                if ((str(id)) not in dict):
                    filename = xlwrite.output('attendance', 'class1', 3, id, 'yes')
                    dict[str(id)] = str(id)

            #if you want to add more ID's then similary add blocks of if else for each ID

        else:
            id = 'Unknown, can not recognize'
            flag = flag + 1
            break

        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
    cv2.imshow('frame', img);
    if flag == 10:
        print("Transaction Blocked")
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

