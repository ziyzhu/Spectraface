import random 
from detect import * 
from eigenface import *

if __name__ == '__main__':

    model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=model_path)
    faces = detector.collect(Spectrum.Thermal, readcache=True, writecache=False)
    persons = detector.create_persons(faces)

    random.shuffle(faces)
    partition = int(len(faces) * 0.75)
    train_faces = faces[:partition]
    test_faces = faces[partition:]

    train_persons = detector.create_persons(train_faces)
    test_persons = detector.create_persons(test_faces)
    recog = EigenfaceRecognizer(train_persons)
    recog.train(train_persons)
    recog.test(test_persons)


