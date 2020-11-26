## About

Achieving state-of-the-art thermal face recognition accuracy with a very simple algorithm using Inception Resnet V1 (pretrained with vggface2) and Eigenface (less than 500 lines of code including optional training script).

Baseline: https://arxiv.org/pdf/1712.02514.pdf

## Downloading Dataset

download from (http://vcipl-okstate.org/pbvs/bench/Data/02/download.html) and place all the downloaded collections under "./dataset".

OR
```
cd thermal-face-recognition/dataset
```
```
wget -r -np -nd -l 1 -A zip http://vcipl-okstate.org/pbvs/bench/Data/02/download.html
```
```
unzip "*.zip" && rm *.zip
```

### Sample thermal and visual face images
![sample thermal image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample1.bmp)
![sample thermal image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample2.bmp)
![sample thermal image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample3.bmp)
![sample visual image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample4.bmp)
![sample visual image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample5.bmp)
![sample visual image](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/sample6.bmp)

## Running

1. ```git clone https://github.com/zachzhu2016/thermal-face-recognition.git```
2. ```(optional) python3 -m venv thermal-face-recognition && source thermal-face-recognition/bin/activate```
3. ```cd thermal-face-recognition```
4. ```pip3 install -r requirements.txt```
5. ```python3 main.py``` (any python3.x except python3.9)

The first run would take about 5 - 7 mintues because it has preprocess all the raw face images. During the first run, face images are detected, cropped, and encoded into a 512 dimension array. The following runs would run within seconds given that preprocessed face images had been cached automatically. 

## Result
![accuracy](https://github.com/zachzhu2016/thermal-face-recognition/blob/main/pictures/accuracy.png)

## Files

- cache.py: pickle utility functions used to store and retrieve preprocessed images
- detect.py: face detection with pretrained model 
- encode.py: using pretrained model to encode detected face images into descriptors fed into the Eigenface algorithm
- eigenface.py: eigenface implementation
- main.py: driver for the program, displays test results

- ./train: used to store training images for fine-tuning 
- ./dataset: contains downloaded dataset for training and testing the algorithm 
- ./cache: contains pickle objects storing preprocessed images
- ./models: contains pretrained thermal face detection model
- ./pictures: contains some insightful plots and sample data

## References
1. Face Recognition: From Traditional to Deep Learning Methods (https://arxiv.org/pdf/1811.00116.pdf)
2. TV-GAN: Generative Adversarial Network Based Thermal to Visible Face Recognition (https://arxiv.org/pdf/1712.02514.pdf)
3. Face Recognition Using Eigenfaces (https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)
4. Eigenfaces for Recognition (https://www.face-rec.org/algorithms/PCA/jcn.pdf)
5. Thermal Infrared Face Recognition – A Biometric Identification Technique for Robust Security system (https://www.intechopen.com/books/reviews-refinements-and-new-ideas-in-face-recognition/thermal-infrared-face-recognition-a-biometric-identification-technique-for-robust-security-system)
6. FaceNet: A Unified Embedding for Face Recognition and Clustering (https://arxiv.org/pdf/1503.03832.pdf)
7. Face Recognition Using Pytorch (https://github.com/timesler/facenet-pytorch)
8. A machine learning model for fast face detection in thermal images (https://github.com/maxbbraun/thermal-face)
