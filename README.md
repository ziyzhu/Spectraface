## About

Achieving state-of-the-art thermal face recognition accuracy with a very simple algorithm using pretrained vggface2 and Eigenface (less than 500 lines of code including training script).

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
- encode.py: using vggface2 to encode detected face images into descriptors fed into the Eigenface algorithm
- eigenface.py: eigenface implementation
- main.py: driver for the program, displays test results
