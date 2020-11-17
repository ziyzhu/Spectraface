from detect import * 
from eigenface import *

if __name__ == '__main__':

    model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=model_path)

    faces = detector.detect(Spectrum.Thermal, readcache=True, writecache=False)

    # persons = detector.create_persons(faces)
    # recog = EigenfaceRecognizer(persons)

    # recog.train()
    # recog.test()
            

