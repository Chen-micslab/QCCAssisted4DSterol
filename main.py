from STAnalyzer import *
import pandas as pd
from emass import emass
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from utils.utils import calculate_k0
from matplotlib import pyplot as plt
from STAnalyzer import STAnalyzer


if __name__ == '__main__':

    myAnalyzer = STAnalyzer()  # create a STAnalyzer

    # myAnalyzer.analyseSingleSTMol(
    #     'CCCC=CC=CCC(=O)O[C@H]1CC[C@@]2(C)C(=CC[C@H]3[C@@H]4CC[C@H]([C@H](C)CCCC(C)C)[C@@]4(C)CC[C@@H]32)C1')
    # # Find all double bond and derivative them.
    # myAnalyzer.STmol.derivativeMolAllPossible(derivativeNumber=(1, 1), chargeNumber=(1, 1), derivativeType=1,
    #                                           useChiral=False)
    # # plot derivatived result
    # myAnalyzer.plotEveryDerivativeMol()
    # myAnalyzer.plotEveryBreakMol()
    # mol = Chem.MolFromSmiles('CCCCC/C=C\C=C\CCC1C([NH+]1C)C/C=C\CCCC(O[C@H]2CC[C@@]3(C)C(C2)=CC[C@H]4[C@@H]5CC[C@H]([C@H](C)CCCC(C)C)[C@@]5(C)CC[C@@H]43)=O')
    # print(Chem.MolToSmiles(mol))


    myAnalyzer.break_NME_mol_from_SMILES('CCCCC/C=C\C=C\CCC1C([NH+]1C)C/C=C\CCCC(O[C@H]2CC[C@@]3(C)C(C2)=CC[C@H]4[C@@H]5CC[C@H]([C@H](C)CCCC(C)C)[C@@]5(C)CC[C@@H]43)=O')
    print('Precusor Mass', myAnalyzer.STmol.noBreakMolMass)
    print('Ring Diagnostic ions',
          myAnalyzer.STmol.diagIonsInRing)  # Delta da 31 is lost derivatived ring mass. Delta da 49 is lost H20 and derivatived ring mass.
    print('Chain Diagnostic ions', myAnalyzer.STmol.diagIonsInChain)
    print('Mol Diagnostic ions', myAnalyzer.STmol.characteristicMassList)
    myAnalyzer.break_NME_mol_from_SMILES(
        'CC1(C([NH+]1C)CC[C@@H](C)[C@H]2CC[C@@]3(C)C4=C(CC[C@]23C)[C@@]5(C)CC[C@H](O)C([C@@H]5CC4)(C)C)C')
    print('Precusor Mass', myAnalyzer.STmol.noBreakMolMass)
    print('Ring Diagnostic ions',
          myAnalyzer.STmol.diagIonsInRing)  # Chain's two fragmentation modes and also consider the lost water result of these molecules.
    print('Chain Diagnostic ions', myAnalyzer.STmol.diagIonsInChain)
    print('Mol Diagnostic ions', myAnalyzer.STmol.characteristicMassList)

