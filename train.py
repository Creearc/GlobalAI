import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import pandas as pd
import numpy as np

import names

df = pd.read_csv('train.csv')
discr_calc = MolecularDescriptorCalculator(names._names)

features = []

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


for elem in df['Smiles']:
    features.append(smi_to_descriptors(elem))

train = pd.DataFrame(data = features , columns=names._names)
train['Active'] = df['Active'].apply(lambda x: 1 if x else 0)

X = train.iloc[:,:-1]
y = train.iloc[:,-1]

print(X)

input_layer = layers.Input(shape=192)
conc = layers.Dense(1, activation='softmax')(input_layer)
model = tf.keras.Model(input_layer, conc)

model.summary()

BATCH_SIZE = 32

EPOCHS = 10
LR = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                      metrics=['accuracy'])

training_dataset = input_fn(features=X.values,
                    labels=y,
                    shuffle=True,
                    num_epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

model.fit(training_dataset,
          steps_per_epoch=len(training_dataset),
          epochs=EPOCHS,
          validation_data=training_dataset,
          validation_steps=len(training_dataset))
