from tqdm import tqdm
import matplotlib.pyplot as plt
import random 

from detect import * 
from eigenface import *
from encode import Encoder

def run_trial(detector, encoder, faces, stranger_ratio=0.2):
    persons = detector.create_persons(faces.copy())
    n_stranger = int(len(persons) * stranger_ratio)
    strangers = random.choices(persons, k=n_stranger)
    stranger_faces = [face for stranger in strangers for face in stranger.faces]
    sample_faces = [face for face in faces if face not in stranger_faces]

    random.shuffle(sample_faces)
    partition = int(len(sample_faces) * 0.80)
    train_faces = sample_faces[:partition]
    test_faces = sample_faces[partition:]

    train_persons = detector.create_persons(train_faces.copy())
    test_persons = detector.create_persons(test_faces.copy())

    recog = EigenfaceRecognizer(train_persons.copy())
    recog.train(verbose=False)

    test_accuracy = recog.test(test_persons.copy())   
    stranger_accuracy = recog.test(strangers.copy()) 
    return {'test_accuracy': test_accuracy, 'stranger_accuracy': stranger_accuracy}

if __name__ == '__main__':

    detection_model_path = 'models/thermal_face_automl_edge_fast.tflite'
    detector = FaceDetector(base_folder='dataset', model_path=detection_model_path)

    tuned_encoder = Encoder('tuned_vggface2')
    encoder = Encoder('vggface2')

    thermal_faces = detector.collect(Spectrum.Thermal, encoder=encoder)
    visual_faces = detector.collect(Spectrum.Visual, encoder=encoder)
    
    datasets = {'thermal_vggface2': {'encoder': encoder, 'faces': thermal_faces.copy()},\
                'thermal_tuned_vggface2': {'encoder': tuned_encoder, 'faces': thermal_faces.copy()},\
                'visual_vggface2': {'encoder': encoder, 'faces': visual_faces.copy()}}
    stranger_ratios = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50] 

    accus = dict()
    for dataset in tqdm(datasets):
        accus[dataset] = dict()
        for stranger_ratio in tqdm(stranger_ratios):
            accus[dataset][stranger_ratio] = run_trial(detector, datasets[dataset]['encoder'], datasets[dataset]['faces'], stranger_ratio)
        for x_a in tqdm(['test_accuracy', 'stranger_accuracy']):
            plt.plot(stranger_ratios, [accus[dataset][stranger_ratio][x_a] for stranger_ratio in accus[dataset]], label=f'{dataset} ({x_a})')

    plt.title = 'Accuracy Plot'
    plt.xlabel('Stranger Ratio (unseen faces / persons)')
    plt.ylabel('Accuracy / 100')
    plt.legend()
    plt.show()

