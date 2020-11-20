from tqdm import tqdm
import random 

from detect import * 
from eigenface import *
from encode import Encoder

if __name__ == '__main__':

    detection_model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=detection_model_path)

    encoder = Encoder('tuned_vggface2')
    faces = detector.collect(Spectrum.Thermal, encoder=encoder, readcache=True, writecache=False)

    persons = detector.create_persons(faces.copy())
    strangers = random.choices(persons, k=3)
    stranger_faces = [face for stranger in strangers for face in stranger.faces]
    sample_faces = [face for face in faces if face not in stranger_faces]

    random.shuffle(sample_faces)
    partition = int(len(sample_faces) * 0.80)
    train_faces = sample_faces[:partition]
    test_faces = sample_faces[partition:]

    train_persons = detector.create_persons(train_faces.copy())
    test_persons = detector.create_persons(test_faces.copy())

    # recog = EigenfaceRecognizer(train_persons.copy())
    # recog.train()

    # recog.test(test_persons.copy()) # test known faces
    # recog.test(strangers.copy())    # test unknown faces



