from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import pandas as pd
import numpy as np

import names

df = pd.read_csv('train.csv')
discr_calc = MolecularDescriptorCalculator(names._names)

features = []

for elem in df['Smiles']:
    features.append(smi_to_descriptors(elem))

print(features)
