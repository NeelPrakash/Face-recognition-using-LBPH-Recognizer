import cv2
import numpy as np

# initiallizing the face detection model

protext_path = "C:/Users/acer/PycharmProjects/Final_Project_Upgraded/deploy.prototxt.txt"
caffemodel_path = "C:/Users/acer/PycharmProjects/Final_Project_Upgraded/res10_300x300_ssd_iter_140000.caffemodel"
face_detection_model = cv2.dnn.readNetFromCaffe(protext_path, caffemodel_path)


# defining the face extractor
def crop_faces(frame):
    (h, w) = frame.shape[:2]

    # get our blob which is our input frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detection_model.setInput(blob)
    detected_faces = face_detection_model.forward()
    cropped_face_coord = list()
    cropped_face_list = list()

    # Iterate over all of the faces detected and extract their start and end points
    for i in range(0, detected_faces.shape[2]):
        box = detected_faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detected_faces[0, 0, i, 2]

        # if the algorithm is more than 16.5% confident that the detection is a face
        if (confidence > 0.175):
            cropped_face_coord.append([startX, startY, endX, endY])

            # sometimes the confidence of network is more then 0.165 percent but face is not present in the frame in such cases
            # "frame[startY:endY, startX:endX]" is an empty list and we cant perform the color conversion on an empty image array
            if len(frame[startY:endY, startX:endX]) != 0:
                cropped_face_list.append(cv2.cvtColor(frame[startY:endY, startX:endX], cv2.COLOR_BGR2GRAY))

    return cropped_face_coord, cropped_face_list

# cv2.rotate(frame[startY:endY, startX:endX],cv2.ROTATE_90_CLOCKWISE)