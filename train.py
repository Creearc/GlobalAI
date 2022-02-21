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

EPOCHS = 10
LR = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                      metrics=['accuracy'])

model.fit(X,
          steps_per_epoch=len(X),
          epochs=EPOCHS,
          validation_data=X,
          validation_steps=len(X))
