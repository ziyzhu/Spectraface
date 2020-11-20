import random 
from detect import * 
from eigenface import *

if __name__ == '__main__':

    model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=model_path)
    faces = detector.collect(Spectrum.Thermal, readcache=True, writecache=False)
    persons = detector.create_persons(faces)

    strangers = random.choices(persons, k=3)
    stranger_faces = [face for stranger in strangers for face in stranger.faces]
    sample_faces = [face for face in faces if face not in stranger_faces]

    random.shuffle(sample_faces)
    partition = int(len(sample_faces) * 0.80)
    train_faces = sample_faces[:partition]
    test_faces = sample_faces[partition:]

    train_persons = detector.create_persons(train_faces)
    test_persons = detector.create_persons(test_faces)
    recog = EigenfaceRecognizer(train_persons)
    recog.train()

    recog.test(test_persons) # test known faces
    recog.test(strangers)    # test unknown faces


