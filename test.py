import cv2
import numpy as np
from keras.models import model_from_json

mapper = {
    0: "angry",
    1: "disgusted",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "suprise"
}

json_file = open('model.json', 'r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)

model.load_weights('model1.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        pred = model.predict(cropped_img)
        max_index = int(np.argmax(pred))
        cv2.putText(frame, mapper[max_index], (x+5, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
