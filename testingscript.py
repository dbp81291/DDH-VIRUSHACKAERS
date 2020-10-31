import csv
import os.path as op
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Torsions
import joblib



#################################TOCOSTUMIZE#################################
file_path = "model.joblib"  #path to model file
smiles = "Clc(c(Cl)c(Cl)c1C(=O)O)c(Cl)c1Cl"

fp_type = 1 #must match the on in the model: 1 == Morgan, 2 == Maccs, 3 == Topological Torsions (or the features you have used in your model)


def load_model(file):
    #load model from pickle file
    if op.exists(file) and op.isfile(file):
        model = joblib.load(open(file, "rb"))
    else:
        print("File does not exist or is corrupted")
        return()
    return(model)

def create_descriptor(smiles, choice):
     #creates fingerprint for given SMILES
    m = AllChem.MolFromSmiles(smiles)
    if choice == 1:
        descriptor = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024)
    elif choice == 2:
        descriptor = MACCSkeys.GenMACCSKeys(m)
    elif choice == 3:
        descriptor = Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m)
    else:
        print ('Invalid fingerprint choice')
    return(descriptor)


if __name__ == '__main__':
    #perform classification for a given SMILES with a saved model

    model = load_model(file_path)
    fp = create_descriptor(smiles, fp_type)
    pred = model.predict(np.reshape(fp, (1, -1)))
    print(pred)
