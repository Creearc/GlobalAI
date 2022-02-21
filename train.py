import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import pandas as pd
import numpy as np

import names
discr_calc = MolecularDescriptorCalculator(names._names)

def smi_to_descriptors(smile):
    mol = Chem.MolFromSmiles(smile)
    descriptors = []
    if mol:
        descriptors = np.array(discr_calc.CalcDescriptors(mol))
    return descriptors


def input_fn(features, labels, shuffle, num_epochs, batch_size):
  """Generates an input function to be used for model training.

  Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training

  Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
  """
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(features))

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


df = pd.read_csv('train.csv')

features = []

for elem in df['Smiles']:
    features.append(smi_to_descriptors(elem))

train = pd.DataFrame(data = features, columns=names._names)
train['Active'] = df['Active'].apply(lambda x: 1 if x else 0)

train = train.sample(frac=1)

l = train.shape[0]
l = int(l * 0.8)

train_X = train.iloc[:l, :-1]
train_y = train.iloc[:l, -1]

eval_X = train.iloc[l:, :-1]
eval_y = train.iloc[l:, -1]

print(train_X)
print(eval_X)

input_layer = layers.Input(shape=192)
conc = layers.Dense(192*2, activation='tanh')(input_layer)
conc = layers.LeakyReLU(0.2)(conc)
conc = layers.Dropout(0.1)(conc)
conc = layers.Dense(192*2, activation='tanh')(conc)
conc = layers.LeakyReLU(0.2)(conc)
conc = layers.Dropout(0.1)(conc)
conc = layers.Dense(1, activation='softmax')(conc)
model = tf.keras.Model(input_layer, conc)

model.summary()

BATCH_SIZE = 32 * 4

EPOCHS = 100
LR = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

training_dataset = input_fn(features=train_X.values,
                    labels=train_y,
                    shuffle=True,
                    num_epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

validation_dataset = input_fn(features=eval_X.values,
                    labels=eval_y,
                    shuffle=False,
                    num_epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

model.fit(training_dataset,
          steps_per_epoch=len(training_dataset),
          epochs=EPOCHS,
          validation_data=validation_dataset,
          validation_steps=len(validation_dataset))
