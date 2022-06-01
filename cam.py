import face_recognition
import cv2
import numpy as np
from time import sleep
import os

# ====================Cascade Verzió==============================
# Cascade-ok útjának megadása                                   #
# face_path = 'haarcascade/haarcascade_frontalface_default.xml'  #
# eye_path = 'haarcascade/haarcascade_eye.xml'                   #
#
# Útvonalak változóba tárolása                                  #
# face_cascade = cv2.CascadeClassifier(face_path)                #
# eye_cascade = cv2.CascadeClassifier(eye_path)                  #
# ================================================================

# A webkamera beolvasása (0 vagy 1, előlapi vagy hátlapi; mikor hogy működik???)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # hibakód elhárításáért kell a 2. paraméter

img_counter = 0
known_face_names = []
take_picture = True
program_running = False

with open('known_names.txt', 'r', encoding='utf-8') as f:
    known_face_names = f.read().splitlines()

print("The names currently stored: ", end="")
if known_face_names:
    print(', '.join(known_face_names))
else:
    print("{empty}")
print("")

# Töröl minden ezelőtti képet vagy folytatja tovább a programot törlés nélkül
delete_pics = input("Do you want to take new pictures and delete the previous ones (if they exist)? (y/n): ")
if delete_pics == "y":
    try:
        for i in range(5):
            os.remove(f'saved_img_{i}.jpg')
            print(f'saved_img_{i} is removed.')
            known_face_names.clear()
            with open('known_names.txt', 'w') as f:
                pass
    except:
        pass
else:
    pass

for i in range(5):
    if os.path.isfile(f'saved_img_{i}.jpg'):
        img_counter = i+1

print("")
while take_picture:
    ret, frame = cap.read()
    cv2.imshow("Picture taker", frame)

    key = cv2.waitKey(30)
    if key == 32:
        if img_counter != 5:
            check, frame = cap.read()
            name = input("Enter who is in the picture: ")
            if name in known_face_names:
                pass
            else:
                known_face_names.append(name)
                with open('known_names.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{name}\n')

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
            print("Program is closing.")
            sleep(5)
            cv2.destroyWindow("Picture taker")
            take_picture = False
            program_running = True

    elif key == 27:
        try:
            path1 = face_recognition.load_image_file('saved_img_0.jpg')
        except FileNotFoundError:
            print("There are no pictures.")
            continue
        print("\nProgram is closing.")
        sleep(1)
        cv2.destroyWindow("Picture taker")
        take_picture = False
        print("Face recognition is starting.")
        program_running = True

image2 = None
image3 = None
image4 = None
image5 = None

image2_face_encoding = None
image3_face_encoding = None
image4_face_encoding = None
image5_face_encoding = None

image1 = face_recognition.load_image_file('saved_img_0.jpg')
image1_face_encoding = face_recognition.face_encodings(image1)[0]

known_face_encodings = [
    image1_face_encoding
]

try:
    image2 = face_recognition.load_image_file('saved_img_1.jpg')
    image2_face_encoding = face_recognition.face_encodings(image2)[0]
    known_face_encodings.append(image2_face_encoding)
except FileNotFoundError:
    pass
try:
    image3 = face_recognition.load_image_file('saved_img_2.jpg')
    image3_face_encoding = face_recognition.face_encodings(image3)[0]
    known_face_encodings.append(image3_face_encoding)
except FileNotFoundError:
    pass
try:
    image4 = face_recognition.load_image_file('saved_img_3.jpg')
    image4_face_encoding = face_recognition.face_encodings(image4)[0]
    known_face_encodings.append(image4_face_encoding)
except FileNotFoundError:
    pass
try:
    image5 = face_recognition.load_image_file('saved_img_4.jpg')
    image5_face_encoding = face_recognition.face_encodings(image5)[0]
    known_face_encodings.append(image5_face_encoding)
except FileNotFoundError:
    pass

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while program_running:
    ret, frame = cap.read()
    if not cap.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    if not ret:
        print("Failed to grab frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame
    # print("Face detected -- {}".format(face_names))

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Az arc köré rajzol egy üres négyzetet
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Az arc fölé rajzol egy teli téglalapot
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

        # A téglalapba helyez a megadott adatok alapján egy nevet
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, color, stroke)

        # # A szemek detektálása egy megadott méretben
        # eye_faces = eye_cascade.detectMultiScale(rgb_small_frame, 1.1, 4)
        # for (px, py, pw, ph) in eye_faces:
        #     #A szemek köré rajzol egy-egy üres négyzetet
        #     cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)

    # Kivetíti azt, amit a webkamera lát pillanatképenként, így olyan lesz mint egy elő videófelvétel
    cv2.imshow('Face recognition', frame)

    # A program folyamatosan fut
    key = cv2.waitKey(30)

    # Leállítja a webkamerát és bezárja az ablakot, ha az ESC billentyű(27) lenyomódott
    if key == 27:
        print("\nTurning off camera.")
        sleep(1)
        cap.release()
        print("Camera off.")
        sleep(1)
        print("Program ended.")
        break

cv2.destroyAllWindows()
