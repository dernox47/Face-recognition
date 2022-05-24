import cv2

# Cascade-ok útjának megadása
face_path = 'haarcascade/haarcascade_frontalface_default.xml'
eye_path = 'haarcascade/haarcascade_eye.xml'

# Útvonalak változóba tárolása
face_cascade = cv2.CascadeClassifier(face_path)
eye_cascade = cv2.CascadeClassifier(eye_path)

# A webkamera beolvasása (0 vagy 1, előlapi vagy hátlapi; mikor hogy működik???)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Arcok detektálása egy megadott méretben, egy koordinátát fog eltárolni a helyzetéről
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Az arc köré rajzol egy üres négyzetet
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        #Az arc fölé rajzol egy teli téglalapot
        cv2.rectangle(frame, (x-1, y), (x+w+1, y-20), (0, 0, 255), cv2.FILLED)

        #A téglalapba helyez a megadott adatok alapján egy nevet
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "unknown"
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (x, y-3), font, 0.7, color, stroke, cv2.LINE_AA)

        #A szemek detektálása egy megadott méretben
        eye_faces = eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (px, py, pw, ph) in eye_faces:
            #A szemek köré rajzol egy-egy üres négyzetet
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)

    #Kivetíti azt, amit a webkamera lát pillanatképenként, így olyan lesz mint egy elő videófelvétel
    cv2.imshow('frame', frame)

    #A program folyamatosan fut, amíg az ESC billentyű(27) nem lesz lenyomva
    key = cv2.waitKey(30)
    if key == 27:
        break

# Leállítja a webkamerát és bezárja az ablakot
cap.release()
cv2.destroyAllWindows()