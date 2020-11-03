import numpy as np
import face_recognition
from adapter import FaceStore, Spectrum, Person, Face

class FaceRecognizer:
    def __init__(self, persons):
        self.persons = persons
        self.mean_face = None
        self.eigfaces = None
        self.eigface_vecs = None
        self.weight_vecs = None

    def __repr__(self):
        return f'FaceRecognizer(persons={len(persons)}, weight_vecs={len(weight_vecs)})'

    def get_mean_face(self, faces):
        width, height = FaceStore.face_shape
        mean_facevec = np.zeros((1, width * height))

        for face in faces:
            facevec = face.image.flatten()
            mean_facevec = np.add(mean_facevec, facevec)

        mean_facevec = np.divide(mean_facevec, float(len(faces))).flatten()
        mean_faceimg = mean_facevec.reshape(FaceStore.face_shape)
        mean_face = Face('meanface', mean_faceimg)
        return mean_face

    def normalize_faces(self, faces):
        normalized_faces = [self.normalize_face(face) for face in faces]
        return normalized_faces

    def normalize_face(self, face):
        normalized_faceimg = np.subtract(face.image, self.mean_face.image)
        normalized_face = Face(face.name, normalized_faceimg) 
        return normalized_face

    def get_train_faces(self):
        train_faces = []
        for p in self.persons:
            face = p.faces[0]
            if face:
                train_faces.append(face)
        return train_faces

    def get_test_faces(self):
        test_faces = []
        for p in self.persons:
            face = p.faces[1]
            if face:
                test_faces.append(face)
        return test_faces

    def train(self):
        train_faces = self.get_train_faces()
        self.mean_face = self.get_mean_face(train_faces)
        normalized_faces = self.normalize_faces(train_faces)
        normalized_facevecs = np.array([face.image.flatten() for face in normalized_faces])

        cov_matrix = np.cov(normalized_facevecs)
        eigvalues, eigvectors = np.linalg.eig(cov_matrix)

        eigpairs = [(eigvalues[i], eigvectors[:, i]) for i in range(len(eigvalues))]
        eigpairs.sort(reverse=True)
        sorted_eigvalues = np.array(list(map(lambda pair: pair[0], eigpairs)))
        sorted_eigvectors = np.array(list(map(lambda pair: pair[1], eigpairs)))

        ncomponents = len(list(filter(lambda p: p < 0.95, np.cumsum(sorted_eigvalues) / sum(sorted_eigvalues))))
        components = np.array(sorted_eigvectors[:ncomponents])
        eigface_vecs = np.dot(components, np.array([f.image.flatten() for f in normalized_faces]))
        eigfaces = [Face(f'eigenface{i}', vec.reshape(FaceStore.face_shape)) for i, vec in enumerate(eigface_vecs)]

        weight_vecs = []
        for face in normalized_faces:
            weight_vec = np.array([np.dot(vec, face.image.flatten()) for vec in eigface_vecs])
            weight_vecs.append(weight_vec)

        self.eigfaces = eigfaces
        self.eigface_vecs = eigface_vecs
        self.weight_vecs = np.array(weight_vecs)

    def test(self):
        test_faces = self.get_test_faces()

    def recognize(face):
        '''
        recognizes a face and returns a person object
        '''
        normalized_face = self.normalize_face(face)
        weight_vec = np.array([np.dot(vec, face.image.flatten()) for vec in self.eigface_vecs])

        return self.persons[0]

if __name__ == '__main__':
    store = FaceStore('./dataset')
    # faces, exceptions = store.detect_faces(Spectrum.Thermal, True)
    faces = store.readcache('faces')
    persons = FaceStore.create_persons(faces)

    recognizer = FaceRecognizer(persons)
    recognizer.train()

    test_faces = recognizer.get_test_faces()
    mean_face = recognizer.get_mean_face(test_faces)
    recognizer.mean_face = mean_face
    normalized_faces = recognizer.normalize_faces(test_faces)
    for face in normalized_faces:
        weight_vec = np.array([np.dot(vec, face.image.flatten()) for vec in recognizer.eigface_vecs])

