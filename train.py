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
