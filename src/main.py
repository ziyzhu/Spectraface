from tqdm import tqdm
import random 

from detect import * 
from eigenface import *
from encode import Encoder

def demo(title, detector, encoder, faces):
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

    recog = EigenfaceRecognizer(train_persons.copy())
    recog.train(verbose=True)

    accuracy_test = recog.test(test_persons.copy())   
    accuracy_stranger = recog.test(strangers.copy()) 
    print()
    print(title)
    print(f'Accuracy for recognizing test faces: {accuracy_test}')
    print(f'Accuracy for recognizing unseen faces: {accuracy_stranger}')

if __name__ == '__main__':

    detection_model_path = '../models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='../dataset', model_path=detection_model_path)

    tuned_encoder = Encoder('tuned_vggface2')
    encoder = Encoder('vggface2')

    thermal_faces = detector.collect(Spectrum.Thermal, encoder=encoder, readcache=True, writecache=False)
    visual_faces = detector.collect(Spectrum.Visual, encoder=encoder, readcache=True, writecache=False)

    demo("Thermal Face Recognition with vggface2: ", detector, encoder, thermal_faces)
    # demo("Thermal Face Recognition with tuned vggface2: ", detector, tuned_encoder, thermal_faces)
    # demo("Visual Face Recognition with vggface2: ", detector, encoder, visual_faces)


