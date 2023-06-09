# Semester II Project

## Face Recognition Attendance System


Automation of the attendance-taking process with a pre-trained face detection and face recognition model. This is capable of capturing images of the face and adding them to a dataset which then can be trained for recognition.

### Features
#### Face detection and Image capturing

The face is recognised using a CAFFE framework model, which is a pre-trained model for object detection. It is associated with Single Shot Multibox Detector(SSD) framework. This is a deep-learning architecture for object detection. 
The option "Add Applicant" enables us to take pictures of individuals and save them under a specified name. Since the model is trained particularly for face detection, it crops out the rest of the area and focuses only on the face thereby creating a much more reliable dataset.

The following image shows how the image capturing will look like.

(IMAGES)![Fce capturing 1](https://github.com/Sn-G-m/Project-Sem_II/assets/133529150/526b9242-a42f-4979-8223-5aa8cd7530d9)
![Face capturing 2](https://github.com/Sn-G-m/Project-Sem_II/assets/133529150/93d1ed87-ea75-41b5-86fb-1989307d5f92)


#### Dataset and Database Creation

The captured image is stored in a folder with the applicant's name. The images are slightly manipulated i.e. brightness and contrasts are varied to increase the accuracy during real-time recognition. An Excel file is also created at the same time (if not already existing) and a sheet with the current month's name is also added to it. 
The names of the applicant will also be entered in the sheet.

An example of the dataset creation and manipulated image folder is shown below.

![Data_set](https://github.com/Sn-G-m/Project-Sem_II/assets/133529150/6996dbef-1e32-47f9-a9b7-45e51a1d711d)

#### Real-Time Recognition and Attendance Marking

After the extraction and training process, the program is now ready for real-time recognition. The face which is in front of the main camera will be recognized and marked as present if the individual's name is in the database. Otherwise, there is an option to add the unrecognised face to the dataset. Attendance will be marked once per person per day.


### Requirements

The code is completely written using Python and the required modules are available in the Requirements.txt file.

### Drawbacks

These are some of the problems which we are facing and would like to work on in the future.

* The dataset is comparatively smaller for a machine learning model to get higher accuracy.
* Each time we add an applicant, training should start from the beginning, therefore time-consuming.
* For that reason, the number of images per person is reduced, which affects the accuracy significantly.
* Proxy attendance cannot be detected.

#### References

+ https://youtu.be/1T1gUsbyMhI
+ [Models and Accuracies - OpenFace (cmusatyalab.github.io)](https://cmusatyalab.github.io/openface/models-and-accuracies/) 
+ [OpenFace (cmusatyalab.github.io)](https://cmusatyalab.github.io/openface/)
+ [Face recognition using OpenCV and Python: A beginner's guide - Blogs - SuperDataScience | Machine Learning | AI | Data Science Career | Analytics | Success](https://www.superdatascience.com/blogs/opencv-face-recognition)


#### Contibutors

| Sl.No  | Name  | Roll.No |
| :------------ |:---------------:| -----:|
| 1      | Sooryanarayanan G | 112201012 |
| 2      | Shreya Verma        |   112201007 |
| 3 | Maloth Sony        |    112201011 |



