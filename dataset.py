import cv2
import numpy as np
import os

extracted_faces = cv2.CascadeClassifier(r"C:\Users\shrey\OneDrive\Desktop\proj_thur\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
    
#to take images for dataset
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #cv2.detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
    faces =extracted_faces.detectMultiScale(gray,1.4,5)
    #returns a list of bounding boxes representing the detected objects in the form of (x, y, w, h), where (x, y) are the coordinates of the top-left corner of the rectangle, and (w, h) are the width and height of the rectangle, respectively.

    #if not faces:
    #if len(faces) > 0:  
    if faces is(): 
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]#storing cropped imgs consisting face(s) detected in input image i.e.'img' here

    return cropped_face


print('------------------------')


name=input('enter name')
pathnew=rf"C:\Users\shrey\OneDrive\Desktop\proj_thur\dataset\{name}"
#new image file named 'pathnew' will be created in the current directory.
if name not in os.listdir(r"C:\Users\shrey\OneDrive\Desktop\proj_thur\dataset"):
    os.mkdir(pathnew)

cap = cv2.VideoCapture(0)#The cap variable is assigned the VideoCapture object, which provides methods for accessing and retrieving video frames from the camera.

count = 0
while True:
    ret, frame = cap.read()#read frames from the camera using the cap.read() method
    #cap.read() returns  ret: a Boolean indicating if the frame was read successfully, and frame: the captured frame.
    
    if face_extractor(frame) is not None:#
        count+=1
        #face=cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
        face = cv2.resize(face_extractor(frame),(500,500))#returns the resized image or frame according to the specified parameters.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    
        file_name_path =rf"C:\Users\shrey\OneDrive\Desktop\proj_thur\dataset\{name}\{name} " +str(count)+'.jpg'
        
        cv2.imwrite(file_name_path,face)
        
        #cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
        cv2.imshow('Face Cropper',face)# creating a window named Fcae copper and displaying face on it

                                                                                    
    """else:

        print("Face not found")
        pass"""

    if cv2.waitKey(1)==13 or count==100:#if enter is pressed or 500 images are taken then it will stop
        break

cap.release()#to release camera from this program
cv2.destroyAllWindows()
print('Samples Colletion Completed ')
