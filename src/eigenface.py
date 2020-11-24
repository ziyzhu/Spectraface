import sys
import numpy as np
import matplotlib.pyplot as plt

from detect import *

class EigenfaceRecognizer:
    DEFAULT_SHAPE = (16, 32) # 16 * 32 = 512 
    def __init__(self, persons):
        self.persons = persons
        self.mean_face = []
        self.eigfaces = []
        self.eigface_vecs = []
        self.weight_vecs = []
        self.logs = []
        
    def __repr__(self):
        return f'EigenfaceRecognizer(persons={len(self.persons)}, weight_vecs={len(self.weight_vecs)})'

    def find_person(self, name):
        for person in self.persons:
            if person.name == name:
                return person
        return None

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

    def recognize(self, face, threshold=0.3, adaptive=False):
        normalized_face = self.normalize_face(face)
        uweight_vec = np.array([np.dot(vec, face.code) for vec in self.eigface_vecs])
        person = None
        mindist = sys.maxsize
        for i, kweight_vec in enumerate(self.weight_vecs):
            dist = np.linalg.norm(kweight_vec - uweight_vec)
            if dist < mindist: 
                person = self.persons[i]
                mindist = dist

        if mindist > threshold and not self.find_person(face.name):
            new_person = Person(face.name)
            new_person.add_face(face)
            self.persons.append(new_person)
            self.train()
            return new_person

        return person

    def train(self, verbose=False):
        train_faces = [self.get_mean_face(p.faces) for p in self.persons] 
        self.mean_face = self.get_mean_face(train_faces)
        normalized_faces = self.normalize_faces(train_faces)
        normalized_facevecs = np.array([face.code for face in normalized_faces])

        cov_matrix = np.cov(normalized_facevecs)
        eigvalues, eigvectors = np.linalg.eig(cov_matrix)

        eigpairs = [(eigvalues[i], eigvectors[:, i]) for i in range(len(eigvalues))]
        eigpairs.sort(reverse=True)
        sorted_eigvalues = np.array(list(map(lambda pair: pair[0], eigpairs)))
        sorted_eigvectors = np.array(list(map(lambda pair: pair[1], eigpairs)))

        components = list(filter(lambda p: p < 0.95, np.cumsum(sorted_eigvalues) / sum(sorted_eigvalues)))
        ncomponents = len(components)
        if verbose == True:
            plt.xlabel('Principal Components')
            plt.ylabel('Cumulative Sum Variance Explained')
            plt.scatter(range(1, ncomponents + 1), components)
            plt.show()

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

    def test(self, test_persons):
        test_faces = [face for p in test_persons for face in p.faces]
        
        logs = []
        correct = 0
        for face in test_faces:
            prediction = self.recognize(face)
            success = prediction.name == face.name
            logs.append({'success': success, 'face': face, 'prediction': prediction})
            if success:
                correct += 1

        accuracy = round(100 * correct / len(test_faces), 3)
        return accuracy
        
        ''' 
        used to determine the threshold value of recognize() function 
        accuracy: 46.341%
        minimum distance for failures: 0.20356772199981246
        average distance for failures: 0.3220990237957095
        maximum distance for successes: 0.443637382783556
        average distance for successes: 0.0011411593799489207
        
        successes = [log for log in logs if log['success'] == True]
        failures = [log for log in logs if log['success'] == False]

        if len(failures) > 0:
            min_fail = min([log['dist'] for log in failures])
            print(f'minimum distance for failures: {min_fail}')

            avg_fail = sum([log['dist'] for log in failures]) / len(failures)
            print(f'average distance for failures: {avg_fail}')
        
        if len(successes) > 0:
            max_success = max([log['dist'] for log in successes])
            print(f'maximum distance for successes: {max_success}')

            avg_success = min([log['dist'] for log in successes]) / len(successes)
            print(f'average distance for successes: {avg_success}')
        '''

