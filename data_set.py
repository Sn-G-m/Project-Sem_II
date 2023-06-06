from PyQt5.QtWidgets import QApplication, QMessageBox
import numpy as np
import datetime
import time
import cv2
import os
from manipulate_image import manipulate


# Specify the file paths for your face detection model
prototxt_path = (
    r"C:\Users\shrey\pythonproj\facial_recognition_attendance_sys"
    r"\face_detection_model\deploy.prototxt"
)
model_path = (
    r"C:\Users\shrey\pythonproj\facial_recognition_attendance_sys"
    r"\face_detection_model\res10_300x300_ssd_iter_140000.caffemodel"
)


# Load your face detection model
extracted_faces = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# Function to extract faces from an image


def face_extractor(img):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    extracted_faces.setInput(blob)
    detections = extracted_faces.forward()
    (h, w) = img.shape[:2]

    faces = []
    for i in range(0, detections.shape[2]):
        detection_score = detections[0, 0, i, 2]
        if detection_score > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Increase the size of the bounding box
            padding = 30  # Adjust this value to increase/decrease the box size
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(w, endX + padding)
            endY = min(h, endY + padding)

            face = img[startY:endY, startX:endX]
            faces.append(face)

    return faces


try:
    # Get user input as name and create a directory for storing the images
    name = input('Enter Name: ')
     
    dataset_path = ( 
        r"C:\Users\shrey\pythonproj"
        r"\facial_recognition_attendance_sys\datasetfolder"
    )
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    name_path = os.path.join(dataset_path, name)
    if not os.path.exists(name_path):
        os.mkdir(name_path)

    cap = cv2.VideoCapture(0)
    count = 0
    dict_names = {}
    while True:
        now = datetime.datetime.now()
        ret, frame = cap.read()
        faces = face_extractor(frame)
        if faces:
            for face in faces:
                if count == 0:
                    time.sleep(2)
                face = cv2.resize(face, (500, 500))
                file_name_path = os.path.join(name_path, f"{name}{count}.jpg")
                dict_names[count] = file_name_path
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(count), (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
                if count < 150:
                    cv2.putText(face,
                                "Face still, eyes on camera",
                                (120, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 130, 0), 2)
                else:
                    if count == 150:
                        time.sleep(2)
                    cv2.putText(face, 
                                "Tilt the face sligtly",
                                (120, 50),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (255, 130, 0), 2)
                cv2.imshow('Face Display', face)
                count += 1
        # when enter key pressed or count reaches 225
        if cv2.waitKey(1) == 13 or count == 225:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Processing data")
    manipulate(dict_names, name_path)
    print('Data Retrieval Complete')

except Exception:
    # Create a QApplication instance
    app = QApplication([])

    # Display an error message box
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle("Error")
    msg_box.setText("Keep your face at some distance")
    msg_box.exec_()

    # Exit the application
    app.quit()


