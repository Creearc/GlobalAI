import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import cv2
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

dataset_path = 'dataset'

model = tf.keras.models.load_model('results/0.9452383649588333__300_3_2_1|1_64_0.1')

labels = [0, 1]
height, width = 300, 300
#height, width = 512, 512

dataset_path = 'test'

counter = 0

f = open('submission.csv', 'w')
t = time.time()
for file in os.listdir(dataset_path):
    #t = time.time()
    img = cv2.imread('{}/{}'.format(dataset_path, file, cv2.IMREAD_COLOR))

    img_n = cv2.resize(img, (width, height))
    img_n = np.expand_dims(img_n, 0)

    y = model.predict_classes(img_n)[0]

    if y == 1:
      counter += 1
    f.write('{}	{}\n'.format(file, y == 1))
    #print(time.time() - t)
f.close()
print(time.time() - t)
print(counter)
