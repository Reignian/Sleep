import os
import pickle
import cv2
import face_recognition
from imutils import paths

def findEncodings(images):
    error_counter = 0
    encodeList = []
    try:
        for img in images:
            img: object = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    except:
        error_counter += 1
        print("Error " + str(error_counter))

    return encodeList

def generate_dataset():
    imagePaths = list(paths.list_images('Dataset'))
    images = []
    classNames = []

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        curImg = cv2.imread(imagePath)
        images.append(curImg)
        classNames.append(imagePath.split(os.path.sep)[-2])

    print(classNames,"test")
    encodeListKnown = findEncodings(images)
    print(encodeListKnown)
    data = {"encodings": encodeListKnown, "names": classNames}
    f = open("face_encodings_final", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print('Encoding Complete Datasets Generated')

generate_dataset()