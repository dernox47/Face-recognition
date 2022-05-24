import cv2
from time import sleep

# Cascade-ok útjának megadása
face_path = 'haarcascade/haarcascade_frontalface_default.xml'
eye_path = 'haarcascade/haarcascade_eye.xml'

# Útvonalak változóba tárolása
face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

# A webkamera beolvasása (0 vagy 1, előlapi vagy hátlapi; mikor hogy működik???)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

img_counter = 0
while True:
    ret, frame = cap.read()
    if not cap.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Arcok detektálása egy megadott méretben, egy koordinátát fog eltárolni a helyzetéről
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Az arc köré rajzol egy üres négyzetet
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Az arc fölé rajzol egy teli téglalapot
        cv2.rectangle(frame, (x-1, y), (x+w+1, y-20), (0, 0, 255), cv2.FILLED)

        # A téglalapba helyez a megadott adatok alapján egy nevet
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "unknown"
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y-3), font, 0.7, color, stroke, cv2.LINE_AA)

        # A szemek detektálása egy megadott méretben
        eye_faces = eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (px, py, pw, ph) in eye_faces:
            #A szemek köré rajzol egy-egy üres négyzetet
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)

    # Kivetíti azt, amit a webkamera lát pillanatképenként, így olyan lesz mint egy elő videófelvétel
    cv2.imshow('Face recognition', frame)

    # A program folyamatosan fut
    key = cv2.waitKey(30)

    # A SPACE megnyomásával készít egy képet, majd megjeleníti és elmenti azt
    if key == 32:
        if img_counter != 5:
            check, frame = cap.read()
            cv2.imwrite(filename=f'saved_img_{img_counter}.jpg', img=frame)
            cap.release()
            cv2.imshow("Captured image", frame)
            cv2.waitKey(2000)
            print("Image saved.")
            img_counter += 1
            cv2.destroyWindow('Captured image')
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            pass
        else:
            print("The maximum number of images is 5.")

    # Leállítja a webkamerát és bezárja az ablakot, ha az ESC billentyű(27) lenyomódott
    elif key == 27:
        print("Turning off camera.")
        cap.release()
        print("Camera off.")
        print("Program ended.")
        break

cv2.destroyAllWindows()
