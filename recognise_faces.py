# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
# import custom function
from FUNCTIONS_FILE import attendance

def push_button_3():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--detector", required=True,
		help="path to OpenCV's deep learning face detector")
	ap.add_argument("-m", "--embedding-model", required=True,
		help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-r", "--recognizer", required=True,
		help="path to model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True,
		help="path to label encoder")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load serialized face detector model
	print("-->Opening face detector")
	# Set path to prototxt file and trained model
	prototxt_path = os.path.sep.join([args["detector"], "deploy.prototxt"])
	model_path = os.path.sep.join([args["detector"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	# Load face detector
	detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

	# load face embedding model
	print("-->Opening face recognizer")
	embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

	# load face recognition model & label encoder
	recognizer = pickle.loads(open(args["recognizer"], "rb").read())
	le = pickle.loads(open(args["le"], "rb").read())

	# initializing the video stream
	print("Streaming started")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	rows = 480
	cols = 640

	# start the FPS 
	fps = FPS().start()

	# loop over frames from video file stream
	while True:
		# take frame from video stream
		frame = vs.read()

		# resize the frame and obtain its dimensions
		frame = imutils.resize(frame, width=640)
		(h, w) = frame.shape[:2]


		# Construct a blob from the image for processing
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		detector.setInput(imageBlob)
		detections = detector.forward()
		if detections.shape[0] == 0:
			print(detections)
			break

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the probability associated with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > args["confidence"]:
				# compute the coordinates of the bounding box for face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face
				face = frame[startY:endY, startX:endX]
				(faceH, faceW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if faceW < 20 or faceH < 20:
					continue

				# Construct a blob from the extracted face region to obtain a
				# 128-dimensional face representation using a face embedding model
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]

				# add attendance of detected person in excel
				attendance(face, name)

				# bounding box around face along with associated probability
				text = f"{name}: {proba * 100:.2f}%"

				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

		# update the FPS counter
		fps.update()
		# show the output frame
		cv2.imshow("Frame", frame)

	    # if the "q" key pressed, break from the loop
		if cv2.waitKey(1) == 13:
			break

	# stop timer and display FPS
	fps.stop()
	print(f"FPS: {fps.fps():.2f}")
	# closing windows
	cv2.destroyAllWindows()
	vs.stop()
