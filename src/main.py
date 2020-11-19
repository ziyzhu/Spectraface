import random 
from detect import * 
from eigenface import *

if __name__ == '__main__':

    model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=model_path)
    faces = detector.collect(Spectrum.Thermal, readcache=True, writecache=False)
    persons = detector.create_persons(faces)

    random.shuffle(faces)
    partition = int(len(faces) * 0.25)
    test_faces = faces[:partition]

    recog = EigenfaceRecognizer(persons)
    test_persons = detector.create_persons(test_faces)
    recog.train(persons)
    recog.test(test_persons)


