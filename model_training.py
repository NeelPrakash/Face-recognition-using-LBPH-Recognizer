import cv2
import os
import numpy
import pickle

# preparing data
dataset_path="C:/Users/acer/PycharmProjects/Final_Project_Upgraded/Database"
if len(os.listdir(dataset_path))!=0:
    (images, lables, names, id) = ([], [], {}, 0)
    for (_, dirs, _) in os.walk(dataset_path):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(dataset_path, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                lable = id
                images.append(cv2.imread(path, 0))
                lables.append(int(lable))
            id += 1

    # saving the list object 'names'
    file = "names.pkl"
    file_obj = open(file,"wb")
    pickle.dump(names, file_obj)
    file_obj.close()

    # Create a Numpy array from the two lists above
    (images, lables) = [numpy.array(li) for li in [images, lables]]

    # initiallizing and training Local Binary Patterns Histogram model
    Face_Recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("Training started....")
    Face_Recognizer.train(images, lables)
    Face_Recognizer.write("C:/Users/acer/PycharmProjects/Final_Project_Upgraded/Face_Recognizer.yml")
    print("Training completed")
else:
    print("oops, Dataset is not present! First create the dataset")