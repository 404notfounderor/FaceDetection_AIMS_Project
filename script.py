#Created and Submitted by Rishik Singh 23/CS/344 for AIMS-DTU internship evaluation task
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []

label_counter=0

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        roi = gray[y:y+h, x:x+w]
        features.append(roi)

        labels.append(label_counter)
        
        cv.rectangle(img,(x,y), (x+w, y+h), (255,0,0), 2)
        label_counter+=1
        cv.putText(img, 'Face', (x,y-4), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA)
    cv.imshow('Face Detection', img)

    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()

print('finished training')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)
