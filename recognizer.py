import cv2
import pickle
from cnn_based_face_extractor import crop_faces

video_object = cv2.VideoCapture(0)

# initiallizing the video_writer_object
fourcc=cv2.VideoWriter_fourcc(*'XVID')
video_writer_obj=cv2.VideoWriter("E:/Pycharm/results2.avi",fourcc,20,(640,480))

# loading the file names.pkl
file = "names.pkl"
file_obj = open(file, "rb")
names = pickle.load(file_obj)

# initiallizing the face recognition model
Face_Recognizer = cv2.face.LBPHFaceRecognizer_create()
Face_Recognizer.read("C:/Users/acer/PycharmProjects/Final_Project_Upgraded/Face_Recognizer.yml")

while video_object.isOpened():
    # frame=cv2.imread("E:/Pycharm/group_photo.jpg")
    _, frame = video_object.read()
    cropped_face_coord, cropped_face_list = crop_faces(frame)

    if len(cropped_face_list)!=0:
        
        # iterating over all of the detected faces
        for i in range(len(cropped_face_list)):
            (startX, startY, endX, endY) = cropped_face_coord[i]
            cv2.rectangle(frame,(startX, startY), (endX, endY),(0, 255, 255),1)

            # recognizing
            id,confidence = Face_Recognizer.predict(cropped_face_list[i])

            if confidence < 100 :
                cv2.putText(frame, names[id], (startX-10, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (24, 255, 0))
            else:
                cv2.putText(frame, "Not Recognized", (startX-10, startY-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 72, 255))

    else:
        cv2.putText(frame,"Face is not detected",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))

    video_writer_obj.write(frame)
    cv2.imshow("Face_recognition",  frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break
video_object.release()
cv2.destroyAllWindows()