# Semester II Project

## Face Recognition Attendance System


Automation of the attendance-taking process with a pre-trained face detection and face recognition model. This is capable of capturing images of the face and adding them to a dataset which then can be trained for recognition.

### Features
#### Face detection and Image capturing

The face is recognised using a CAFFE framework model, which is a pre-trained model for object detection. It is associated with Single Shot Multibox Detector(SSD) framework. This is a deep-learning architecture for object detection. 
The option "Add Applicant" enables us to take pictures of individuals and save them under a specified name. Since the model is trained particularly for face detection, it crops out the rest of the area and focuses only on the face thereby creating a much more reliable dataset.

The following image shows how the image capturing will look like.

(IMAGES)

#### Dataset and Database Creation

The captured image is stored in a folder with the applicant's name. The images are slightly manipulated i.e. brightness and contrasts are varied to increase the accuracy during real-time recognition. An Excel file is also created at the same time (if not already existing) and a sheet with the current month's name is also added to it. 
The names of the applicant will also be entered in the sheet.

(IMAGES)

#### Real-Time Recognition and Attendance Marking

After the extraction and training process, the program is now ready for real-time recognition. The face which is in front of the main camera will be recognized and marked as present if the individual's name is in the database. Otherwise, there is an option to add the unrecognised face to the dataset. Attendance will be marked once per person per day.

(IMAGES)

### Requirements

The code is completely written using Python and the required modules are available in the Requirements.txt file.

### Drawbacks

These are some of the problems which we are facing and would like to work on in the future.

* The dataset is comparatively smaller for a machine learning model to get higher accuracy.
* Each time we add an applicant, training should start from the beginning, therefore time-consuming.
* For that reason, the number of images per person is reduced, which affects the accuracy significantly.
* Proxy attendance cannot be detected.







