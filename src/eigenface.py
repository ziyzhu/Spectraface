import sys
import numpy as np
from detect import *

class EigenfaceRecognizer:
    DEFAULT_SHAPE = (16, 32) # 16 * 32 = 512 
    def __init__(self, persons):
        self.persons = persons
        self.mean_face = []
        self.eigfaces = []
        self.eigface_vecs = []
        self.weight_vecs = []

    def __repr__(self):
        return f'EigenfaceRecognizer(persons={len(self.persons)}, weight_vecs={len(self.weight_vecs)})'

    def get_mean_face(self, faces):
        width, height = self.DEFAULT_SHAPE
        mean_facevec = np.zeros((1, width * height))

        for face in faces:
            facevec = face.code
            mean_facevec = np.add(mean_facevec, facevec)

        mean_facevec = np.divide(mean_facevec, float(len(faces))).flatten()
        mean_faceimg = mean_facevec.reshape(self.DEFAULT_SHAPE)
        mean_face = Face('meanface', image=mean_faceimg, code=mean_facevec)
        return mean_face

    def normalize_faces(self, faces):
        normalized_faces = [self.normalize_face(face) for face in faces]
        return normalized_faces

    def normalize_face(self, face):
        normalized_facevec = np.subtract(face.code, self.mean_face.code)
        normalized_face = Face(name=face.name, code=normalized_facevec) 
        return normalized_face

    def get_train_faces(self):
        train_faces = []
        for p in self.persons:
            face = self.get_mean_face(p.faces)
            if face:
                train_faces.append(face)
        return train_faces

    def get_test_faces(self):
        test_faces = []
        for i, p in enumerate(self.persons):
            if i > 0:
                test_faces.extend(p.faces)
        return test_faces

    def recognize(self, face):
        normalized_face = self.normalize_face(face)
        uweight_vec = np.array([np.dot(vec, face.code) for vec in self.eigface_vecs])
        person = None
        mindist = sys.maxsize
        for i, kweight_vec in enumerate(self.weight_vecs):
            dist = np.linalg.norm(kweight_vec - uweight_vec)
            if dist < mindist: 
                person = self.persons[i]
                mindist = dist
        return person

    def train(self):
        train_faces = self.get_train_faces()
        self.mean_face = self.get_mean_face(train_faces)
        normalized_faces = self.normalize_faces(train_faces)
        normalized_facevecs = np.array([face.code for face in normalized_faces])

        cov_matrix = np.cov(normalized_facevecs)
        eigvalues, eigvectors = np.linalg.eig(cov_matrix)

        eigpairs = [(eigvalues[i], eigvectors[:, i]) for i in range(len(eigvalues))]
        eigpairs.sort(reverse=True)
        sorted_eigvalues = np.array(list(map(lambda pair: pair[0], eigpairs)))
        sorted_eigvectors = np.array(list(map(lambda pair: pair[1], eigpairs)))

        ncomponents = len(list(filter(lambda p: p < 0.95, np.cumsum(sorted_eigvalues) / sum(sorted_eigvalues))))
        components = np.array(sorted_eigvectors[:ncomponents])
        eigface_vecs = np.dot(components, np.array([face.code for face in normalized_faces]))
        eigfaces = [Face(f'eigenface{i}', vec.reshape(self.DEFAULT_SHAPE)) for i, vec in enumerate(eigface_vecs)]

        weight_vecs = []
        for face in normalized_faces:
            weight_vec = np.array([np.dot(vec, face.code) for vec in eigface_vecs])
            weight_vecs.append(weight_vec)

        self.eigfaces = eigfaces
        self.eigface_vecs = eigface_vecs
        self.weight_vecs = np.array(weight_vecs)

    def test(self):
        test_faces = self.get_test_faces()
        predictions = []
        for face in test_faces:
            person = self.recognize(face)
            predictions.append(person.name)

        correct = 0
        for face, prediction in zip(test_faces, predictions): 
            if face.name == prediction: 
                correct += 1

        accuracy = round(100 * correct / len(test_faces), 3)
        print(f'Test Results: accuracy={accuracy}%')
        return accuracy

