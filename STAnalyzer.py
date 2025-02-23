import pandas as pd
import numpy as np
import fingerPrint
from emass.emass import EMass
from typing import List, Union, Tuple
from rdkit.Chem.rdchem import Mol
from STMol import STMolecule
from rdkit import Chem
from tqdm import tqdm
from plot import showMolWithIdx, newShowMol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


# def STRingsPD(STAnalysePath: str):
#     stAnalysePd = pd.read_csv('ST_Analyse.csv')
#     stWithRingPd = stAnalysePd.loc[stAnalysePd['Standard'] == 'YES']
#     stWithoutRingPd = stAnalysePd.loc[stAnalysePd['Standard'] == 'NO']
#     stWithRingPd.to_csv('ST_WithRing_Analyse.csv')
#     stWithoutRingPd.to_csv('ST_WithoutRing_Analyse.csv')


class STAnalyzer:

    def __init__(self):
        self.STWithDoubleBondPd = object
        self.STWithDoubleBondPdCopy: pd.DataFrame = None
        self.molNumber = int
        self.STmol: STMolecule = None

    def readCSV(self, filePath: str):
        self.STWithDoubleBondPd = pd.read_csv(filePath, index_col=0)  #
        self.STWithDoubleBondPdCopy = self.STWithDoubleBondPd

    def analyseSingleSTMol(self, mol: Union[Mol, str], useChiral: bool = True):
        self.STmol = STMolecule(mol, useChiral=useChiral)

    def analyseAllSTMol(self, savepath=None):
        with tqdm(total=len(self.STWithDoubleBondPd), dynamic_ncols=True) as bar:
            for index, row in self.STWithDoubleBondPd.iterrows():
                LMST = row['LM_ID']
                mol = Chem.MolFromSmiles(row['SMILES'])
                self.STmol = STMolecule(mol)
                self.STWithDoubleBondPdCopy.loc[index, 'rotatablePercent'] = self.STmol.rotatablePercent
                self.STWithDoubleBondPdCopy.loc[index, 'oxygenNumber'] = str(self.STmol.oxygenNumber)
                self.STWithDoubleBondPdCopy.loc[index, 'doublebondNumber'] = str(self.STmol.doubleBondNumber)
                self.STWithDoubleBondPdCopy.loc[index, 'OHBondNumber'] = str(self.STmol.OHBondNumber)
                self.STWithDoubleBondPdCopy.loc[index, 'doublebondNumberCanBreak'] = str(len(self.STmol.doubleBondBrokenIdx))
                if self.STmol.doubleBondBrokenIdx:
                    self.STWithDoubleBondPdCopy.loc[index, 'DoubleBondInChain'] = 'Yes'
                else:
                    self.STWithDoubleBondPdCopy.loc[index, 'DoubleBondInChain'] = 'No'
                if self.STmol.isSugar:
                    self.STWithDoubleBondPdCopy.loc[index, 'isSugar'] = 'Yes'
                else:
                    self.STWithDoubleBondPdCopy.loc[index, 'isSugar'] = 'No'
                bar.set_description_str(desc=f'Now analyse {LMST}')
                bar.update(1)
        self.STWithDoubleBondPdCopy.drop(['SYSTEMATIC_NAME', 'CATEGORY', 'INCHI_KEY', 'INCHI',
                                          'ABBREVIATION', 'PUBCHEM_CID', 'SYNONYMS', 'HMDB_ID',
                                          "CHEBI_ID", "SWISSLIPIDS_ID", "KEGG_ID", "LIPIDBANK_ID"],
                                         axis=1, inplace=True)
        # self.STWithDoubleBondPdCopy = self.STWithDoubleBondPdCopy[
        #     self.STWithDoubleBondPdCopy['MAIN_CLASS'] != 'Steroid conjugates [ST05]']
        self.STWithDoubleBondPdCopy = self.STWithDoubleBondPdCopy[self.STWithDoubleBondPdCopy['EXACT_MASS'] <= 900]
        if savepath:
            self.writeCSV(filePath=savepath)
        else:
            self.writeCSV(filePath='./database/AllSTAnalyseResult.csv')

    def getSingleDerivationSmiles(self, filePath: str, sameCCW: int = 0, derivativeNumber: int = 1, charge: int = 1,
                                  derivativeType: int = 1,
                                  chiral: List[int] = None, protonationChiral: List[int] = None):
        """
        :param filePath:
        :param sameCCW: 0 diff CCW,1 same CCW
        :param derivativeNumber: add derivative mol number
        :param charge:
        :param derivativeType: 1 TS-H,2 TS-Me
        :param chiral:0,1 represent different chiral
        :param protonationChiral:0,1 represent different protonation chiral
        :return:
        """
        derivativeAllMolDict = {}
        derivativeAllMolDict.update(
            self.STmol.generateDerivationSmilesFile(molName='test', sameCCW=sameCCW, filePath=filePath,
                                                    derivativeNumber=derivativeNumber, charge=charge,
                                                    derivativeType=derivativeType,
                                                    chiral=chiral,
                                                    protonationChiral=protonationChiral))
        derivativeAllMolPd = pd.DataFrame.from_dict(derivativeAllMolDict, orient='index', columns=['SMILES'])
        derivativeAllMolPd.index.name = 'LMID'
        try:
            derivativeAllMolPd.to_csv(filePath)
        except:
            filePath = filePath.split('.')[0] + 'TS-H_Smiles_No_CCW.csv'
            derivativeAllMolPd.to_csv(filePath)
        return None

    def getAllDerivativeSmiles(self, savePath: str, sameCCW: int = 0, derivativeNumber: Tuple[int, int] = (1, 1),
                               chargeNumber: Tuple[int, int] = (1, 1),
                               derivativeType: int = 1, useChiral: bool = True,
                               chiral: List[int] = None, protonationChiral: List[int] = None):
        """
        :param useChiral:
        :param chargeNumber:
        :param savePath: derivated mol csv file save path
        :param sameCCW: 0 diff CCW,1 same CCW
        :param derivativeNumber: add derivative mol number
        :param derivativeType: 1 TS-Me,2 TS-H
        :param chiral:0,1 represent different chiral
        :param protonationChiral:0,1 represent different protonation chiral
        :return:
        """
        derivativeAllMolDict = {}
        length = len(self.STWithDoubleBondPd)
        with tqdm(total=length, dynamic_ncols=True) as bar:
            for index, row in self.STWithDoubleBondPd.iterrows():
                LM_ID = row['LM_ID']
                smiles = row['SMILES']
                self.STmol = STMolecule(smiles)
                derivativeAllMolDict.update(
                    self.STmol.generateDerivationSmiles(molName=LM_ID, sameCCW=sameCCW,
                                                        derivativeNumber=derivativeNumber,
                                                        chargeNumber=chargeNumber, derivativeType=derivativeType,
                                                        chiral=chiral, protonationChiral=protonationChiral,
                                                        useChiral=useChiral)
                )
                bar.set_description_str(desc=f'Now derivative {LM_ID}')
                bar.update(1)
            derivativeAllMolPd = pd.DataFrame.from_dict(derivativeAllMolDict, orient='index', columns=['SMILES'])
            derivativeAllMolPd.index.name = 'LMID'
            try:
                derivativeAllMolPd.to_csv(savePath)
            except:
                savePath = savePath.split('.')[0] + 'TS-H_Smiles_No_CCW.csv'
                derivativeAllMolPd.to_csv(savePath)
        return None

    def getAllDerivativeChainBreakMass(self, filePath: str, savePath: str = 'MSMS/MS2.csv', OHMaxBreak: int = None):
        df = pd.read_csv(filePath)
        length = len(df)
        resultDict = {}
        myemass = EMass()
        with tqdm(total=length, dynamic_ncols=True) as bar:
            for index, row in df.iterrows():
                mol = Chem.MolFromSmiles(row['SMILES'])
                LM_ID = row['LMID']
                formula = CalcMolFormula(mol).split('+')[0]
                mz = round(myemass.calculate(formula=formula, limit=0.001, charge=1)[0].mass, 4)
                self.break_NME_mol_from_SMILES(mol, OHMaxBreak=OHMaxBreak)
                diagIonsInChian = [str(round(x,4)) for x in self.STmol.diagIonsInChain]
                diagIonsInRing = [str(round(x,4)) for x in self.STmol.diagIonsInRing]
                characteristicMass = [str(round(x,4)) for x in self.STmol.characteristicMassList]
                diagIonsInChianStr = ','.join(diagIonsInChian)
                diagIonsInRingStr = ','.join(diagIonsInRing)
                characteristicMassStr = ','.join(characteristicMass)
                resultDict[LM_ID] = [mz, diagIonsInChianStr, diagIonsInRingStr, characteristicMassStr]
                bar.set_description_str(desc=f'Now fragment {LM_ID}')
                bar.update(1)
            derivativeAllMolPd = pd.DataFrame.from_dict(resultDict, orient='index',
                                                        columns=['mz', 'diagIonsInChian', 'diagIonsInRing',
                                                                 'characteristicMass'])
            derivativeAllMolPd.round(4)
            derivativeAllMolPd.to_csv(savePath)

    def getBreakCharacteristicMass(self, lowDectectionMs: float = 100):
        self.STmol.fragDerivation()  # fragment function
        self.STmol.calBreakCharacteristicMass(lowMsDectection=lowDectectionMs)  # calculate fragment ions
        # print(self.STmol.characteristicMassList)

    def break_NME_mol_from_SMILES(self, smiles: str, OHMaxBreak: int = 2):
        self.STmol = STMolecule(smiles, useChiral=True)
        self.STmol.getDerivativeMolFromSmiles(OHMaxBreak=OHMaxBreak)
        # if not self.STmol.getDerivativeMolFromSmiles():
        #     print('No break ring')
        # else:
        #     print(self.STmol.characteristicMassList)

    def generateAllDerivationMolFingerPrint(self, derivativeCSV: str, savePath: str = None):
        fPName = fingerPrint.getMordredDescriptorsName()
        derivativePd = pd.read_csv(derivativeCSV)
        header = True
        with tqdm(total=len(derivativePd), dynamic_ncols=True) as bar:
            for index, data in derivativePd.iterrows():
                name = data["LMID"]
                smiles = data["SMILES"]
                fPvalues = fingerPrint.getFPUseMordred(Chem.MolFromSmiles(smiles))
                fPDict = dict(zip(fPName, fPvalues))
                allMolFingerPrintPd = pd.DataFrame(fPDict, index=[name])
                allMolFingerPrintPd.index.name = 'LMID'
                if header:
                    allMolFingerPrintPd.to_csv(savePath, mode='a', header=header)
                    header = False
                else:
                    allMolFingerPrintPd.to_csv(savePath, mode='a', header=header)
                bar.set_description_str(desc=f'Now Get {name}\'s fingerPrint')
                bar.update(1)

    @staticmethod
    def contactAndDropFingerPrintDuplicates(rdkitFpPath: str, rcdkFpPath: str, savePath: str):
        rdkitFpPD = pd.read_csv(rdkitFpPath, low_memory=False)
        rcdkFpPath = pd.read_csv(rcdkFpPath, low_memory=False)
        fpPd = pd.concat((rdkitFpPD, rcdkFpPath), axis=1)
        fpPd = fpPd.iloc[:, 1:]
        fpPd.set_index('LMID', inplace=True)
        fpPd = fpPd.replace([np.inf, -np.inf], np.nan)
        fpPd.dropna(axis=1, inplace=True)
        unUniqueIndex = fpPd.apply(lambda x: x.nunique(), axis=0).eq(1)  # 单个分子描述符重复
        # fp_duplicated_columns = fpPd.columns[unUniqueIndex]  # 分子描述符之间重复项
        fpPdReserved = fpPd.T.loc[~unUniqueIndex.values].T
        fpPdReserved = fpPdReserved.T.drop_duplicates().T
        fpPdReserved.to_csv(savePath)


    @staticmethod
    def dropFingerPrintDuplicates(rdkitFpPath: str, savePath: str):
        rdkitFpPD = pd.read_csv(rdkitFpPath, low_memory=False,index_col=0)
        rdkitFpPD = rdkitFpPD.iloc[:, :]
        fpPd = rdkitFpPD.replace([np.inf, -np.inf], np.nan)
        fpPd.dropna(axis=1, inplace=True)
        unUniqueIndex = fpPd.apply(lambda x: x.nunique(), axis=0).eq(1)  # 单个分子描述符重复
        # fp_duplicated_columns = fpPd.columns[unUniqueIndex]  # 分子描述符之间重复项
        fpPdReserved = fpPd.T.loc[~unUniqueIndex.values].T
        fpPdReserved = fpPdReserved.T.drop_duplicates().T
        fpPdReserved = fpPdReserved.round(2)
        fpPdReserved.to_csv(savePath)

    def plotEveryDerivativeMol(self, save: bool = False, showBond: bool = False, showAtom: bool = False):
        for key in self.STmol.derivativeMolDict.keys():
            titleStr = ''
            if self.STmol.derivativeMolDict[key]['aziridineOnRingAtomIdxDict']:
                titleStr += 'ring loction idx' + str(
                    next(iter(self.STmol.derivativeMolDict[key]['aziridineOnRingAtomIdxDict'])))
            if self.STmol.derivativeMolDict[key]['aziridineOnChainAtomIdxDict']:
                titleStr += 'chain loction idx' + str(
                    next(iter(self.STmol.derivativeMolDict[key]['aziridineOnChainAtomIdxDict'])))
            newShowMol(key, title=titleStr, rotateDegree=157)
            if save:
                showMolWithIdx(key, showAtom=showAtom, showBond=showBond)

    def plotEveryBreakMol(self):
        for mol in self.STmol.derivativeMolDict.keys():
            newShowMol(mol, title="Precursor ion", rotateDegree=157)
            self.STmol.fragDerivation()
            for mol_ in self.STmol.afterDerivationBreakMolDict[mol]:
                newShowMol(mol_, title="Product ion", rotateDegree=157)
                print(Chem.MolToSmiles(mol_))

    def writeCSV(self, filePath: str):
        self.STWithDoubleBondPdCopy.to_csv(filePath)

    def statisticsSterol(self, filePath: str):
        df = pd.read_csv(filePath)
        isomerNumber = df.duplicated('FORMULA', keep=False).sum()
        doubleBondNumber = len(df)
        nonIsomerNumber = len(df) - isomerNumber
        withChainDoubleBondNumber = len(df.loc[df['DoubleBondInChain'] == 'Yes'])
        withoutChainDoubleBondNumber = len(df.loc[df['DoubleBondInChain'] == 'No'])
        print(f'Sterol with double bond number {doubleBondNumber}')
        print(f'Isomer with double bond: {isomerNumber} percent: {isomerNumber/doubleBondNumber:.1%}')
        print(f'Isomer without double bond: {nonIsomerNumber} percent: {nonIsomerNumber/doubleBondNumber:.1%}')
        print(f'With chain: {withChainDoubleBondNumber} percent: {withChainDoubleBondNumber / doubleBondNumber:.1%}')
        print(f'Without chain: {withoutChainDoubleBondNumber} percent: {withoutChainDoubleBondNumber / doubleBondNumber:.1%}')
