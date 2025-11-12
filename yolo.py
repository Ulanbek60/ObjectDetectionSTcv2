import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('animals.mp4')

if not cap:
    print('Camera not found')
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print('Картина жок')
        break
    result = model(frame, stream=True)

    for i in result:
        for n in i.boxes:
            x, y , w, h = map(int, n.xyxy[0])

            conf = float(n.conf[0])
            cls = int(n.cls[0])
            label = model.names[cls]

            if conf < 0.5:
                continue

            cv2.rectangle(frame, (x,y), (w, h), (0, 255, 255), 2)
            cv2.putText(frame, f'{label}{conf:2f}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),2)

    cv2.imshow('FRAME WITH OBJECTS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
