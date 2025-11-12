import cv2
import datetime

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Ошибка загрузки каскадов!")
    exit()

detect_eyes = True
detect_smile = True
save_faces = False
face_counter = 0

print("Детектор лиц запущен!")
print(" [e] — вкл/выкл детекцию глаз")
print(" [s] — вкл/выкл детекцию улыбки")
print(" [c] — сделать снимок")
print(" [a] — автосохранение лиц")
print(" [q] — выход")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтения кадра.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        if detect_eyes:
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                radius = int((ew + eh) / 4)
                cv2.circle(frame, eye_center, radius, color=(0, 255, 0), thickness=2)

        if detect_smile:
            roi_smile = roi_gray[int(h / 2):h, :]

            smiles = smile_cascade.detectMultiScale(
                roi_smile,
                scaleFactor=1.8,
                minNeighbors=20,
                minSize=(25, 25)
            )

            if len(smiles) > 0:
                cv2.putText(frame, "SMILE :)", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if save_faces:
            face_img = frame[y:y + h, x:x + w]
            filename = f"face_{face_counter}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, face_img)
            face_counter += 1

    info_text = f"Faces: {len(faces)}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    status_y = 60
    if detect_eyes:
        cv2.putText(frame, "Eyes: ON", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        status_y += 25

    if detect_smile:
        cv2.putText(frame, "Smile: ON", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        status_y += 25

    if save_faces:
        cv2.putText(frame, "Auto-save: ON", (10, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    cv2.imshow(winname='Face Detection', mat=frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Завершение работы.")
        break
    elif key == ord('e'):
        detect_eyes = not detect_eyes
        print(f"Детекция глаз: {'ВКЛ' if detect_eyes else 'ВЫКЛ'}")
    elif key == ord('s'):
        detect_smile = not detect_smile
        print(f"Детекция улыбки: {'ВКЛ' if detect_smile else 'ВЫКЛ'}")
    elif key == ord('c'):
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Скриншот сохранен: {filename}")
    elif key == ord('a'):
        save_faces = not save_faces
        if not save_faces:
            face_counter = 0
        print(f"Автосохранение: {'ВКЛ' if save_faces else 'ВЫКЛ'}")

cap.release()
cv2.destroyAllWindows()

print(f"Всего обработано лиц: {face_counter}")
