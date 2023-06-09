"""
importing modules
"""
from imutils import paths     # Basic image processing functions
import imutils
import numpy as num           # For processing and manipulation of data
import argparse               # For command-line interface
import pickle                 # For serialising and de-serialising
from cv2 import dnn                   # An opencv module
import os                     # Provides a way to interact with the OS
import cv2
from train_data import train
# File paths
def push_button_2():
    DEFAULT_DATASET = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                    r"\SemIIProject\Sample2\Final_Project\dataset")
    DEFAULT_VECTOR = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                    r"\SemIIProject\Sample2\Final_Project\output"
                    r"\ExtractedData.PICKLE")
    DEFAULT_DETECTOR = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                        r"\SemIIProject\Sample2\Final_Project"
                        r"\face_detection_model")
    DEFAULT_MODEL = (r"C:\Users\DELL\AppData\Local\Programs\Python\Python311"
                    r"\SemIIProject\Sample2\Final_Project"
                    r"\openface_nn4.small2.v1.t7")
    DEFAULT_CONFIDENCE = 0.75


    # Adding argument
    arg_parser = argparse.ArgumentParser()   # Creating an ArgumentParser instance

    arg_parser.add_argument("-ds", "--data_set", default=DEFAULT_DATASET)
    arg_parser.add_argument("-vf", "--vector_file", default=DEFAULT_VECTOR)
    arg_parser.add_argument("-fd", "--face_detector", default=DEFAULT_DETECTOR)
    arg_parser.add_argument("-md", "--model", default=DEFAULT_MODEL)
    arg_parser.add_argument("-cl", "--confidence_level",
                            default=DEFAULT_CONFIDENCE, type=float)

    # Converting namespace object to dictionary
    arg_dict = vars(arg_parser.parse_args())

    # Retrieve face detection model from disk

    # Path for network architecture defining file
    proto_file_path = os.path.sep.join([arg_dict["face_detector"],
                                        "model_architecture.prototxt"])
    # Path for pre-trained model for face detection
    model_path = os.path.sep.join([arg_dict["face_detector"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])

    # Reading and iniitializing the detector using Caffe framework
    face_detector = dnn.readNetFromCaffe(proto_file_path, model_path)
    print("<< Face detector loading completed successfully >>")

    # Retrieve face embedding model from disk
    # Path of embedding model
    emb_model_path = arg_dict["model"]

    # Reading and iniitializing the embedding model using Torch framework
    embedding_model = dnn.readNetFromTorch(emb_model_path)
    print("<< Face recognising model loading completed successfully >>")

    # Image location for training as a list
    data = arg_dict["data_set"]
    img_path = list(paths.list_images(data))
    print("<< Image location retrieving >>")

    # Container for data
    facial_embs = []
    face_names = []

    N_FACES = 0             # Number of proper images

    # Iterating through the files
    for (serial_no, location) in enumerate(img_path):
        print("Processing Image, iteratiion {}/{}".format((serial_no + 1),
                                                        len(img_path)))
        person = location.split(os.path.sep)[-2]

        # Loading images
        image = cv2.imread(location)
        image = imutils.resize(image, width=600)
        (height, width) = image.shape[: 2]

        # Constructing a vector dimensional array of the image
        img_tensor = dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0),
                                        swapRB=False, crop=False)

        # Locating face with Deep Neural Network Model
        face_detector.setInput(img_tensor)
        # Output
        detect_output = face_detector.forward()

        if len(detect_output) > 0:             # Atleast one face detection
            # Getting the bounding boxes and confidence level
            index = num.argmax(detect_output[0, 0, :, 2])
            confidence = detect_output[0, 0, index, 2]
            if confidence >= arg_dict["confidence_level"]:
                # Getting coordinates and re-scaling
                limits = detect_output[0, 0, index, 3: 7] * num.array(
                    [width, height, width, height])
                # Pixel positions
                (x_1, y_1, x_2, y_2) = limits.astype("int")

                face = image[y_1: y_2, x_1: x_2]
                face_height, face_width = face.shape[: 2]

                if face_height < 20 or face_width < 20:
                    continue

                face_tensor = cv2.dnn.blobFromImage(face, 1.0/255,
                                                    (96, 96), (0, 0, 0),
                                                    swapRB=True, crop=False)
                embedding_model.setInput(face_tensor)
                vector = embedding_model.forward()

                face_names.append(person)
                facial_embs.append(vector.flatten())

                N_FACES += 1

    # Saving data to a file
    face_data = {"vector_file": facial_embs, "names": face_names}
    file = open(arg_dict["vector_file"], "wb")
    file.write(pickle.dumps(face_data))
    file.close()
    train()