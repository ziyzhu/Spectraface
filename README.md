## About

Achieving state-of-the-art thermal face recognition accuracy with a very simple algorithm using Inception Resnet V1 (pretrained with vggface2) and Eigenface (less than 500 lines of code including optional training script).

Baseline: https://arxiv.org/pdf/1712.02514.pdf

## Downloading Dataset:

1. download from: http://vcipl-okstate.org/pbvs/bench/Data/02/download.html
2. place all the downloaded collections under "./dataset".

## Running:

1. (optional) python3 -m venv thermal-face-recognition
2. pip3 install -r requirements.txt
3. python3 main.py

## Files

- cache.py: pickle utility functions used to store and retrieve preprocessed images
- detect.py: face detection
- encode.py: using pretrained model to encode detected face images into descriptors fed into the Eigenface algorithm
- eigenface.py: eigenface implementation
- main.py: driver for the program, displays test results

- ./train: used to store training images for fine-tuning 
- ./dataset: contains downloaded dataset
- ./cache: contains pickle objects storing preprocessed images
- ./models: contains pretrained thermal face detection model
- ./src: implementation 
- ./pictures: contains some insightful plots 

## References
1. Face Recognition: From Traditional to Deep Learning Methods (https://arxiv.org/pdf/1811.00116.pdf)
2. TV-GAN: Generative Adversarial Network Based Thermal to Visible Face Recognition (https://arxiv.org/pdf/1712.02514.pdf)
3. Face Recognition Using Eigenfaces (https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)
4. Eigenfaces for Recognition (https://www.face-rec.org/algorithms/PCA/jcn.pdf)
5. Thermal Infrared Face Recognition – A Biometric Identification Technique for Robust Security system (https://www.intechopen.com/books/reviews-refinements-and-new-ideas-in-face-recognition/thermal-infrared-face-recognition-a-biometric-identification-technique-for-robust-security-system)
6. FaceNet: A Unified Embedding for Face Recognition and Clustering (https://arxiv.org/pdf/1503.03832.pdf)
7. Face Recognition Using Pytorch (https://github.com/timesler/facenet-pytorch)
