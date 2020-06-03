import cv2
import os
import imutils
from cnn_based_face_extractor import crop_faces

video_object = cv2.VideoCapture("C:/Users/acer/PycharmProjects/sunil_sir1.mp4")
cap_frames = 0

path="C:/Users/acer/PycharmProjects/Final_Project_Upgraded/Database"
os.chdir(path)

# folder name must matches with the person name

sub_Name = input("Enter Subject Name: ")
while os.path.exists(sub_Name):
    print("This subject already exists, Please enter a new subject name!")
    sub_Name = input("Enter Subject Name: ")
os.mkdir(sub_Name)

while video_object.isOpened():
    _, frame = video_object.read()
    _, cropped_face_list = crop_faces(frame)

    if len(cropped_face_list)==1:
        croped_face = cropped_face_list[0].copy()

        # saving the cropped images into the apropriate folder
        file_path = "C:/Users/acer/PycharmProjects/Final_Project_Upgraded/Database/" + str(sub_Name) + "/img" + str(cap_frames) + ".jpg"
        cv2.imwrite(file_path, croped_face)
        cap_frames += 1

        cv2.putText(croped_face, "Captured_frames: " + str(cap_frames), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))
        cv2.imshow("cropped_face",imutils.resize(croped_face, width=200))

    elif len(cropped_face_list)>1:
        cv2.putText(frame, "more then one face is found", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1)

    else:
        cv2.putText(frame, "Face_not_detected", (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 1)

    cv2.putText(frame, "captured_frames: " + str(cap_frames), (450, 50), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255))
    cv2.imshow("creating_database", frame)

    if cv2.waitKey(1) & 0xFF == 32 or cap_frames==411:
        break

video_object.release()
cv2.destroyAllWindows()
