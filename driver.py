import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition

INVALID_NAMES = {'meng2', '.DS_Store', 'DISGUISE'}
FACE_SIZE = (129, 129)

class Spectrum:
    Thermal = 'Thermal'
    Visual = 'Visual'
    List = [Thermal, Visual]

class Expression:
    Surprised = 'ex1'
    Laughing = 'ex2'
    Angry = 'ex3'
    List = [Surprised, Laughing, Angry]

class Illumination:
    BothLightsOn = '2on'
    DarkRoom = 'Dark'
    LeftLightOn = 'Lon'
    LightsOff = 'Off'
    RightLightOn = 'Ron'
    List = [BothLightsOn, DarkRoom, LeftLightOn, LightsOff, RightLightOn]

class Person:
    def __init__(self, name):
        self.name = name
        self.faces = []

    def add_face(self, face):
        self.faces.append(face)

    def show_faces(self):
        for face in self.faces:
            face.show()

    def __repr__(self):
        return f'Person(name={self.name}, faces={len(self.faces)})'

class Face:
    def __init__(self, name, faceimg, filepath = None, spec: Spectrum = None, exp: Expression = None, illmt: Illumination = None):
        self.name = name
        self.image = faceimg
        self.spectrum = spec
        self.expression = exp
        self.illumination = illmt
        self.filepath = filepath

    def show(self):
        plt.imshow(self.image)
        plt.title(self.name)
        plt.show()

    def __repr__(self):
        return f'Face(name={self.name}, spectrum={self.spectrum}, expression={self.expression}, illumination={self.illumination})'

class FaceStore:

    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.names = [fname for fname in os.listdir(f'{base_folder}') if fname not in INVALID_NAMES]

    def cache(self, objname, obj):
        with open(f'./cache/{objname}.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def read(self, objname):
        try: 
            with open(f'./cache/{objname}.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def find_facefiles(self, name, spec: Spectrum = None, exp: Expression = None, illmt: Illumination = None) -> str:
        path = None
        if exp:
            path = f'{self.base_folder}/{name}/Expression/{exp}' 
        elif illmt:
            path = f'{self.base_folder}/{name}/Illumination/{illmt}' 
        else:
            raise 

        facefiles = []
        try:
            facefiles = os.listdir(path)
        except:
            return []

        if spec == Spectrum.Thermal:
            facefiles = list(filter(lambda fname: fname[0] == 'L', facefiles))
        elif spec == Spectrum.Visual:
            facefiles = list(filter(lambda fname: fname[0] == 'V', facefiles))

        return [f'{path}/{fname}' for fname in facefiles if '.bmp' in fname]

    def detect_faces(self, spec, cache = False):
        '''
        returns a list of face objects containing cropped face images
        About 30% of faces are detected from the original dataset
        '''

        faces = []
        exceptions = []
        for name in store.names[:1]:
            for exp, illmt in zip(Expression.List, Illumination.List):
                facefiles_exp = self.find_facefiles(name, spec=spec, exp=exp, illmt=None)
                facefiles_illmt = self.find_facefiles(name, spec=spec, exp=None, illmt=illmt)
                facefiles = facefiles_exp + facefiles_illmt
                for f in facefiles:
                    try: 
                        img = face_recognition.load_image_file(f)
                        facelocs = face_recognition.face_locations(img)
                        if facelocs:
                            # 1. crop 
                            top, right, bottom, left = facelocs[0]
                            faceimg = cv2.resize(img[top:bottom, left:right], FACE_SIZE)
                            # 2. to gray scale
                            faceimg = cv2.cvtColor(faceimg, cv2.COLOR_BGR2GRAY)

                            face = Face(name, faceimg, f, spec, exp, illmt)
                            faces.append(face)

                    except Exception as e:
                        exceptions.append(e)
                        continue

        if cache:
            self.cache('faces', faces)

        return faces, exceptions

class FaceRecognizer:
    def __init__(self, persons):
        self.persons = persons

    def mean_face(self, faces):
        width, height = FACE_SIZE
        mean_facevec = np.zeros((1, width * height))

        for name, face in faces.items():
            facevec = face.image.flatten()
            mean_facevec = np.add(mean_facevec, facevec)

        mean_facevec = np.divide(mean_facevec, float(len(faces))).flatten()
        mean_faceimg = mean_facevec.reshape(width, height)
        mean_face = Face(name, mean_faceimg)

        return mean_face

    def train_faces(self):
        train_faces = {}
        for name, p in persons.items():
            train_face = p.faces[0] # temp
            if train_face:
                train_faces[train_face.name] = train_face

        return train_faces

    def train(self):
        train_faces = self.train_faces()
        mean_face = self.mean_face(train_faces)

    def recognize(face):
        '''
        recognizes a face and returns a person object
        '''
        pass

store = FaceStore('./dataset')
# faces, exceptions = store.detect_faces(Spectrum.Thermal, True)
faces = store.read('faces')
persons = {}
for face in faces:
    if persons.get(face.name):
        persons[face.name].add_face(face)
    else:
        person = Person(face.name)
        person.add_face(face)
        persons[face.name] = person

recognizer = FaceRecognizer(persons)
mean_face = recognizer.mean_face()

