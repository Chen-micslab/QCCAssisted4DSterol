from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import numpy as np


def getMordredDescriptorsName():
    calc = Calculator(descriptors, ignore_3D=True)
    # a = str(calc.descriptors[0].as_argument)
    name = list(map(lambda x: x.to_json()['name'] + '_' + str(x.as_argument), list(calc.descriptors)))
    return ['molwt'] + name


def getFPUseMordred(mol):
    calc = Calculator(descriptors, ignore_3D=True)
    molWt = Descriptors.MolWt(mol)
    fp = list(calc(mol))
    fp.insert(0, molWt)
    fpr = []
    for x in fp:
        try:
            value = np.float64(x)
        except:
            value = np.nan
        fpr.append(value)
    return fpr


def listFingerPrintIntoDict(fingerPrint: list, desciptorDict: dict, fingerPrintName: str):
    i = 1
    for x in fingerPrint:
        name = fingerPrintName + '_' + str(i)
        desciptorDict[name] = x
        i += 1
    return desciptorDict

