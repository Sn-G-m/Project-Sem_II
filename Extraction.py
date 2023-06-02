# importing modules
from imutils import paths     # Basic image processing functions
import imutils
import numpy as num           # For processing and manipulation of data
import argparse               # For command-line interface
import pickle                 # For serialising and de-serialising a python object structure
import cv2                    # An opencv interface for face detection and recogniton
import os                     # Provides a way to interact with the operating system

# Adding argument
arg_parser = argparse.ArgumentParser()        # Creating an instance of ArgumentParser

arg_parser.add_argument("-ds", "--data_set", required = True)
arg_parser.add_argument("-vf", "--vector_file", required = True)
arg_parser.add_argument("-fd", "--face_detector", required = True)
arg_parser.add_argument("-md", "--model", required = True )
arg_parser.add_argument("-cl", "--confidence_level", type = float, default = 0.75)

arg_dict = vars(arg_parser.parse_args())      # Converting namespace object to dictionary


# Retrieve face detection model from disk
proto_file_path = os.path.sep.join([arg_dict["face_detector"], "model_architecture.prototxt"])              # Path for network architecture defining file
model_path = os.path.sep.join([arg_dict["face_detector"], "res10_300x300_ssd_iter_140000.caffemodel"])      # Path for pre-trained model for face detection

face_detector = cv2.dnn.readNetFromCaffe(proto_file_path, model_path)                                       # Reading and iniitializing the detector using Caffe framework


# Retrieve face embedding model from disk
emb_model_path = arg_dict["model"]                                   # Path of embedding model

embedding_model = cv2.dnn.readNetFromTorch(emb_model_path)           # Reading and iniitializing the embedding model using Torch framework

# Image location for training as a list
data = arg_dict("data_set")
img_path = list(paths.list_images(data))



