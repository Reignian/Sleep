import cv2
import face_recognition
import pickle
import numpy as np

data = pickle.loads(open('face_encodings_final', "rb").read())
print(str(data["encodings"]) + "Test")

cap = cv2.VideoCapture(0)
counter = 0
recognizeCounter = 0
holdCurrentStud = ""



while True:
    success, img = cap.read()
    if len(data["encodings"]):
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(data["encodings"], encodeFace)
            faceDis = face_recognition.face_distance(data["encodings"], encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = data["names"][matchIndex]
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name.upper(), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 2)

                cv2.putText(img, name + '(Present)', (50, 50), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0), 2)
            else:
                cv2.putText(img, '(No data)', (50, 50), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0), 2)

    cv2.namedWindow('WebCam', cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow('WebCam', img)
    if cv2.waitKey(1) & 0xFF == ord('|'):
        break
cap.release()
cv2.destroyAllWindows()