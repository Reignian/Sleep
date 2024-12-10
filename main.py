import cv2
# opencv-contrib-python

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")
def captureImgSample():
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = faceCascade.detectMultiScale(gray_img, 1.1, 10)
    coordinate = []
    check = 0

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Face", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        coordinate = [x, y, w, h]

    print(coordinate)
    if len(coordinate) == 4:
        roi_img = img[coordinate[1]:coordinate[1] + coordinate[3], coordinate[0]:coordinate[0] + coordinate[2]]
        user_id = 1
        check = 1
        cv2.imwrite("Dataset/Joevin/Image." + str(user_id) + "." + str(img_id) + ".jpg", roi_img)
    return check

img_id =1
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    check = captureImgSample()
    cv2.imshow("Testing", img)

    print(img_id)
    if check == 1:
        img_id += 1

    if img_id == 51:
        break

    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()