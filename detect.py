import os 
import numpy as np
import tflite_runtime.interpreter as tflite
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import cache
from encode import Encoder

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
    def __init__(self, name=None, image=None, code=None, filepath=None, spec: Spectrum=None, exp: Expression=None, illmt: Illumination=None):
        self.name = name
        self.image = image
        self.spectrum = spec
        self.expression = exp
        self.illumination = illmt
        self.filepath = filepath
        self.code = code

    def encode(self):
        return f'{self.name}_{self.spectrum}_{self.expression}_{self.illumination}'

    def show(self):
        self.image.show(title=self.name)

    def __repr__(self):
        return f'Face(name={self.name}, spectrum={self.spectrum}, expression={self.expression}, illumination={self.illumination})'
    
    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        face = Face()
        face.__dict__.update(d)
        return face

class FaceDetector:
    def __init__(self, base_folder, model_path):
        self.base_folder = base_folder
        invalid_names = {'meng2', '.DS_Store', 'DISGUISE'}
        self.names = [fname for fname in os.listdir(f'{base_folder}') if fname not in invalid_names]

        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.interpreter = interpreter

    def create_persons(self, faces):
        persons = []
        for face in faces:
            person = None 
            for p in persons:
                if p.name == face.name:
                    person = p
            if person:
                person.add_face(face)
            else:
                person = Person(face.name)
                person.add_face(face)
                persons.append(person)
        return persons

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
    
    def detect(self, faces):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        floating_model = input_details[0]['dtype'] == np.float32
        input_height = input_details[0]['shape'][1]
        input_width = input_details[0]['shape'][2]
                
        for face in tqdm(faces):
            img = Image.open(face.filepath).resize((input_width, input_height))
            input_data = np.expand_dims(img, axis=0)
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(output_details[0]['index'])
            labels = self.interpreter.get_tensor(output_details[1]['index'])
            confidences = self.interpreter.get_tensor(output_details[2]['index'])
            nboxes = self.interpreter.get_tensor(output_details[3]['index'])

            maxindex = np.argmax(confidences[0])
            box = boxes[0][maxindex] * 192 # width == height == 192px
            box = [box[1], box[0], box[3], box[2]]

            # (optional)
            # draw = ImageDraw.Draw(img)
            # draw.rectangle(box, outline="red")

            cropped = img.crop(box)
            face.image = cropped
        return faces

    def collect(self, spec=Spectrum.Thermal, encoder=Encoder('vggface2'), readcache=True, writecache=False):
        cached_file = cache.findcache(f'faces_{spec}_{encoder.name}')
        if readcache and cached_file:
            dicts = cache.readcache(f'faces_{spec}_{encoder.name}')
            faces = [Face.from_dict(d) for d in dicts]
            return faces

        faces = []
        for name in self.names:
            for exp, illmt in zip(Expression.List, Illumination.List):
                facefiles_exp = self.find_facefiles(name, spec=spec, exp=exp, illmt=None)
                facefiles_illmt = self.find_facefiles(name, spec=spec, exp=None, illmt=illmt)
                facefiles = facefiles_exp + facefiles_illmt
                for f in facefiles:
                    faces.append(Face(name=name, filepath=f, spec=spec, exp=exp, illmt=illmt))
        
        faces = self.detect(faces)
        for face in tqdm(faces):
            face.code = encoder.encode(face.image)

        if writecache or not cached_file:
            for face in faces: 
                face.image = np.array(face.image)
            dict_list = [face.to_dict() for face in faces]
            cache.writecache(f'faces_{spec}_{encoder.name}', dict_list)

        return faces

