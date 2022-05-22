import cv2

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.rectangle(frame, (x-1, y), (x+w+1, y-20), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "unknown"
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y-3), font, 0.7, color, stroke, cv2.LINE_AA)

        eye_faces = eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (px, py, pw, ph) in eye_faces:
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()