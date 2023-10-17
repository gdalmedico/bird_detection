import cv2
from datetime import datetime
import numpy as np

cap = cv2.VideoCapture(0)

#variaveis para detecção de movimento
_, start_frame = cap.read();
start_frame = start_frame[90:390, 170:470]
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)
alarm = False
alarm_mode = False
alarm_counter = 0
alarm_pessoa = False

#variaveis para detecção de objetos
prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.1
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def analisa_imagem(image):
    #image = image[90:390, 170:470]
    print("Analisando a imagem...")
    tem_ave = False
    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()
    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            if classes[class_index]=='bird':
                tem_ave = True
                upper_left_x = int((detected_objects[0, 0, i, 3]) * width)
                upper_left_y = int((detected_objects[0, 0, i, 4]) * height)
                lower_right_x = int((detected_objects[0, 0, i, 5]) * width)
                lower_right_y = int((detected_objects[0, 0, i, 6]) * height)
                prediction_text = f"{classes[class_index]}: {confidence:.2f}%"
                cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
                cv2.putText(image, prediction_text,
                            (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    global alarm
    if tem_ave:
        print("PASSARO DETECTADO")
        cv2.imshow("Detected objects", image)
        salva_imagem_passaro(image)
        salva_imagem(image)
        alarm = False
    else:
        print("APENAS MOVIMENTO")
        salva_imagem(image)
        alarm = False

def salva_imagem(frame):
    exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
    local_image="imalertas/mov/alert" + str(exact_time) + ".jpg"
    cv2.imwrite(local_image, frame)

def salva_imagem_passaro(frame):
    exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
    local_image="imalertas/det/alert" + str(exact_time) + ".jpg"
    cv2.imwrite(local_image, frame)

while True:
    ret, frame = cap.read()
    frame = frame[90:390, 170:470]
    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (21, 21), 0)
        difference = cv2.absdiff(frame_bw, start_frame)
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        start_frame = frame_bw
        if threshold.sum() > 50:
            print(threshold.sum())
            alarm_counter += 1
        else:
            if alarm_counter > 0:
                alarm_counter -= 1
        cv2.imshow("Cam", threshold)
    else:
        cv2.imshow("Cam", frame)

    if alarm_counter > 3:
        if not alarm:
            alarm = True
            if alarm_pessoa:
                analisa_imagem(frame)
            else:
                alarm = False

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode
        alarm_counter = 0
    if key_pressed == ord("q"):
        alarm_mode = False
        break
    if key_pressed == ord("p"):
        alarm_pessoa = not alarm_pessoa
        alarm_counter = 0

cap.release()
cv2.destroyAllWindows()
