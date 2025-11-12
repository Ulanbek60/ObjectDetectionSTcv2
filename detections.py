import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

print("🎥 Детектор движения запущен")
print(" [s] — сделать снимок движения")
print(" [r] — сброс фона")
print(" [q] — выход")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

sensitivity = 40 # 0-255
min_area = 500

while True:
    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    _, thresh = cv2.threshold(src=blur, thresh=sensitivity, maxval=255, type=cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, kernel=None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    display_frame = frame1.copy()

    motion_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Area: {int(area)}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    status = "Motion detected!" if motion_detected else "No motion"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)

    cv2.putText(display_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(display_frame, f"Sensitivity: {sensitivity}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow(winname='Motion detector', mat=display_frame)
    cv2.imshow(winname='Threshold (debugging)', mat=thresh)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("🛑 Завершение работы.")
        break
    elif key == ord('s'):
        filename = f"motion_{cv2.getTickCount()}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"📷 Снимок сохранен: {filename}")
    elif key == ord('r'):
        print("🔄 Сброс фонового кадра")
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        continue
    elif key == ord('+') or key == ord('='):
        sensitivity = min(sensitivity + 5, 100)
        print(f"Чувствительность: {sensitivity}")
    elif key == ord('-'):
        sensitivity = max(sensitivity - 5, 5)
        print(f"Чувствительность: {sensitivity}")

    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        print("Ошибка чтения кадра.")
        break

cap.release()
cv2.destroyAllWindows()
