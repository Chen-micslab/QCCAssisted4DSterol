# encoding: utf-8
# by SunJian
# This class used for frag ST MOL based on Rdkit. Use GetSubstructMatches function find location to . Next version will change based on
# version 0.1.5

""" Module containing the core chemistry functionality of the lipidccs """
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import Mol, Bond  # Mol is a class
from typing import List, Union, Dict, Tuple
import template
import copy
import itertools
from emass.emass import EMass
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcNumRotatableBonds
import numpy as np
from plot import newShowMol


class radicalGroupSmart:
    # oxhydryl bond
    oxhydrylBondSmart: str = '[O;H1]-[C]-[C;!H0]'
    oxhydrylBondOnRingSmart: str = '[O;H1]-[C&R]-[C;!H0]'
    oxhydrylBondOnChainSmart: str = '[O;H1]-[C!R]-[C;!H0]'
    anyOxhydrylBondSmart: str = '[O;H1]-[C]'
    # double bond
    doubleBondInRingSmart: str = '[C&R]=[C&R]'
    doubleBondBetweenRingAndChainSmart: str = '[C&R]=[C&!R]'
    doubleBondInChainSmart: str = '[C&!R]=[C&!R]'
    # aziridine ring
    aziridineRingSmart: str = '[C]-[N]1-[C&R]-[C&R]-1'
    aziridineBreakSmart: str = '[C]-[N]1-[C]-[C]-1'
    # CE bond
    CE_esterBondBrokenSmart: str = '[#6&R]-[#8]-[#6](-[#6]-[#6])=[#8]'
    # sugar bond
    sugarSmart: str = '[#8]-[#6@@H]1-[#6]-[#6@H](-[#8])-[#8]-[#6]-[#6@H]-1-[#8]'


class STMolecule:
    def __init__(self, mol: Union[Mol, str], useChiral:bool=True):
        """
            derivativeMolDict: key:
            Rdkit mol value:[Aziridines N index,(Aziridines C index1,Aziridines C index2)]
        :param mol:
        """
        self.mol: Union[Mol, str, type(None)] = mol  # Warning: Need deep copy in some usage.
        self.oxygenNumber: int = 0
        # ----------OHBond------------
        self.OHBond: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.OHOnRingBond: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.OHOnChainBond: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.OHBondNumber: int = 0
        # ----------DoubleBond--------
        self.doubleBond: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.doubleBondInRing: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.doubleBondInChain: Dict[int, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.doubleBondBetweenRingAndChain: Dict[
            Bond, Tuple[int, int]] = {}  # Dict(Bond Index: Tuple(atomIdx1,atomIdx2))
        self.doubleBondBrokenIdx: List[int] = []  # Double bond not in ring index list
        self.doubleBondNumber: int = 0
        self.doubleBondLocInSTRingList = None
        # ----------EsterBond-------
        self.CEBond: Dict[int, Tuple[int, int]] = {}
        # ----------SugarBond-------
        self.isSugar = False
        # ----------reaction site--------
        self.reactionSiteList: List[tuple] = []  # List[tuple(reactionDoubleBondAtom1Idx,reactionDoubleBondAtom2Idx))]
        self.reactionChargeSiteDict: Dict[tuple, List[List[int]]]  # {reactionDoubleBondIdx:list[list[chargeOnBondIdx]]}
        self.derivativeMolDict: Dict[
            RWMol, Dict[str, Dict[int, List[int]]], Dict[str, Dict[int, List[int]]], int, int] = {}
        """{After derivative mol RWMol:{
         'aziridineOnRingAtomIdxDict':{double bond site idx on ring:[aziridineRingNIdx,aziridineRingCIdx,
                                        aziridineRingDoubleBondAtom1Idx,aziridineRingDoubleBondAtom2Idx]},
         'aziridineOnChainAtomIdxDict':{double bond site idx on chain:[aziridineRingNIdx,aziridineRingCIdx,
                                        aziridineRingDoubleBondAtom1Idx,aziridineRingDoubleBondAtom2Idx]},
         'chiral':int
         'rotonationChiral':int
         } """
        self.afterDerivationBreakMolDict: Dict[RWMol, List[RWMol]] = {}
        self._templateUnspecifiedRingsMol: Mol = template.tempRingUnspecifiedMol
        self._templateSingleRingsMol: Mol = template.tempRingSingleMol
        self._locationTemplateMolDict: dict = template.temRingLocMolDict
        self.bondFragmentIdxList: List[tuple] = []
        self.isStandard: bool = False
        self.reactionSiteList: List[tuple] = []
        self.isStandard: bool = False
        self.isCE: bool = False
        self.cantbreak = False
        self.rotatablePercent: float = 0.0
        self.molSmarts: str = ''
        self.molSmiles: str = ''

        self.diagIonsInChain: List[float] = []
        self.diagIonsInRing: Dict[str:List[float], str:List[float]] = {}
        self.characteristicMassList: List[int] = []
        self.noBreakMolMass:float = None
        self._initialFunction(useChiral)
        self._analyseMol()

    def _initialFunction(self, useChiral):
        self.getMolFromSmiles(self.mol, useChiral)
        self.RWMol = RWMol(self.mol)
        self.formula = CalcMolFormula(self.mol)

    def _analyseMol(self):
        self._getOxygenNumber()
        self._findOHBond()
        self._findDoubleBond()
        self._findCELikeBond()
        self._findSugarBond()
        self._judgeStandardST()
        self._calRotatableRatio()
        self.findDoubleBondLocationInSTRings()

    def getMolFromSmiles(self, mol: Union[Mol, str], useChiral):
        """
            get molecule from smiles or Mol
        :param mol: Chem Mol object or SMILES str
        """
        if not useChiral:
            if isinstance(mol, str):
                self.mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=False))
                self.molSmarts = Chem.MolToSmarts(self.mol)
                self.molSmiles = mol
            elif isinstance(mol, Mol):
                self.mol: Mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
                self.molSmarts = Chem.MolToSmarts(self.mol)
                self.molSmiles = Chem.MolToSmiles(self.mol)
        else:
            if isinstance(mol, str):
                self.mol = Chem.MolFromSmiles(mol)
                self.molSmarts = Chem.MolToSmarts(self.mol)
                self.molSmiles = mol
            elif isinstance(mol, Mol):
                self.mol: Mol = mol
                self.molSmarts = Chem.MolToSmarts(self.mol)
                self.molSmiles = Chem.MolToSmiles(self.mol)

    def _getOxygenNumber(self):
        """
            Get oxygen atom number
        """
        ONumber = 0
        for atom in self.mol.GetAtoms():
            if 8 == atom.GetAtomicNum():
                ONumber += 1
        self.oxygenNumber = ONumber

    def _findCELikeBond(self):
        """
            Find CE like bond index in mol
        """
        self.isCE = True
        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.CE_esterBondBrokenSmart))
        for (a, b, c, e, f, g) in bondAtomStartEndTuple:
            self.CEBond[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)
        self.CEBondNumber = len(self.CEBond)

    def _findSugarBond(self):
        bondAtomStartEndTuple = self.mol.GetSubstructMatches(Chem.MolFromSmarts(radicalGroupSmart.sugarSmart))
        if len(bondAtomStartEndTuple) > 0:
            self.isSugar = True

    def _findOHBond(self):
        """
            Find OH bond index in mol
        """
        bondAtomStartEndTuple = self.mol.GetSubstructMatches(Chem.MolFromSmarts(radicalGroupSmart.oxhydrylBondSmart))
        for (a, b, c) in bondAtomStartEndTuple:
            self.OHBond[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)
        self.OHBondNumber = len(self.OHBond)

        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.oxhydrylBondOnRingSmart))
        for (a, b, c) in bondAtomStartEndTuple:
            self.OHOnRingBond[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)

        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.oxhydrylBondOnChainSmart))
        for (a, b, c) in bondAtomStartEndTuple:
            self.OHOnChainBond[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)

    def _findDoubleBond(self):
        """
            Find double bond index in mol
        """
        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.doubleBondInRingSmart))
        for (a, b) in bondAtomStartEndTuple:
            self.doubleBondInRing[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)
        # bondAtomStartEndTuple = self.mol.GetSubstructMatches(
        #     Chem.MolFromSmarts(radicalGroupSmart.doubleBondInChainSmart))
        # for (a, b, c, d) in bondAtomStartEndTuple:
        #     self.doubleBondInChain[self.mol.GetBondBetweenAtoms(b, c).GetIdx()] = (b, c)
        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.doubleBondInChainSmart))
        for (a, b) in bondAtomStartEndTuple:
            self.doubleBondInChain[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)
        # bondAtomStartEndTuple = self.mol.GetSubstructMatches(
        #     Chem.MolFromSmarts(radicalGroupSmart.doubleBondBetweenRingAndChainSmart))
        # for (a, b, c, d) in bondAtomStartEndTuple:
        #     self.doubleBondBetweenRingAndChain[self.mol.GetBondBetweenAtoms(b, c).GetIdx()] = (b, c)

        bondAtomStartEndTuple = self.mol.GetSubstructMatches(
            Chem.MolFromSmarts(radicalGroupSmart.doubleBondBetweenRingAndChainSmart))
        for (a, b) in bondAtomStartEndTuple:
            self.doubleBondBetweenRingAndChain[self.mol.GetBondBetweenAtoms(a, b).GetIdx()] = (a, b)
        self.doubleBond.update(self.doubleBondBetweenRingAndChain)
        self.doubleBond.update(self.doubleBondInChain)
        self.doubleBond.update(self.doubleBondInRing)
        self.doubleBondNumber = len(self.doubleBond)
        self.doubleBondBrokenIdx += list(self.doubleBondInChain.keys())
        self.doubleBondBrokenIdx += list(self.doubleBondBetweenRingAndChain.keys())

    def _calRotatableRatio(self):
        rotatableBondsNumber = CalcNumRotatableBonds(self.mol)
        removeHMol = Chem.RemoveAllHs(self.mol)
        allBondsNumber = removeHMol.GetNumBonds()
        self.rotatablePercent = rotatableBondsNumber / allBondsNumber

    def _judgeStandardST(self):
        """
            Judge a mol is a molecule with template ring.
        """
        res = rdFMCS.FindMCS([self._templateSingleRingsMol, self.mol], bondCompare=rdFMCS.BondCompare.CompareAny)
        resMol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(resMol)
        resMolRingInfo = resMol.GetRingInfo()
        if resMolRingInfo.NumRings() != 4:
            self.isStandard = False
            # self.mol.SetProp('MolType', 'Not_Standard_ST')
        else:
            self.isStandard = True
            # self.mol.SetProp('MolType', molType.standardST)

    def _combinationsReactionAndChargeBondSite(self, derivativeNumber: Tuple[int, int] = None,
                                               chargeNumber: Tuple[int, int] = None):
        """
        This function can enumerate the combination of the number of derivatives and charge sites,
        in order to provide combinations parameter of the derivative function.
        :param derivativeNumber:(derivative number min, derivative number max)
        :param chargeNumber:(charge number min, charge number max)
        """
        self.reactionChargeSiteDict = {}
        if derivativeNumber[0] < 1:
            print(f'Derivation Number is less than 0. No Derivation')
        # if chargeNumber[0] < 1:
        #     print(f'Charge Number is less than 0. Mass Spectrometer can not dectect this mol')
        if derivativeNumber[0] > self.doubleBondNumber:
            print(f'Derivation number min must less than double bond number {self.doubleBondNumber}')
        if chargeNumber[1] - 1 > derivativeNumber[0]:
            raise Exception(f'Molecule charge number max must less than derivative number {derivativeNumber}')
        derivativeNumberRange = None
        chargeNumberRange = None
        if derivativeNumber:
            derivativeNumberRange = range(derivativeNumber[0], derivativeNumber[1] + 1)
        if chargeNumber:
            chargeNumberRange = range(chargeNumber[0], chargeNumber[1] + 1)
        if derivativeNumber is None and chargeNumber is None:
            derivativeNumberRange = range(0, self.doubleBondNumber + 1)
            chargeNumberRange = range(0, self.doubleBondNumber + 1)
        for repeatNumber in derivativeNumberRange:
            for x in list(itertools.combinations(self.doubleBond.keys(), repeatNumber)):
                self.reactionSiteList.append(x)
        for reactionSite in self.reactionSiteList:
            chargeSiteList = []
            for repeatNumber in chargeNumberRange:
                for chargeSiteTuple in itertools.combinations(reactionSite, r=repeatNumber):
                    chargeSiteList.append(list(chargeSiteTuple))
            self.reactionChargeSiteDict[reactionSite] = chargeSiteList

    def getDerivativeMolFromSmiles(self, lowMsDetection: float = 100, OHMaxBreak: int = 2):
        aziridineRingTuple = self.mol.GetSubstructMatches(Chem.MolFromSmarts(radicalGroupSmart.aziridineBreakSmart))[0]
        (CH3AtomIdx, NAtomIdx, doubleBondAtom1Idx, doubleBondAtom2Idx) = aziridineRingTuple
        doubleBond = self.mol.GetBondBetweenAtoms(doubleBondAtom1Idx, doubleBondAtom2Idx)
        breakBondList = [self.mol.GetBondBetweenAtoms(doubleBondAtom1Idx, NAtomIdx).GetIdx(),
                         self.mol.GetBondBetweenAtoms(doubleBondAtom2Idx, NAtomIdx).GetIdx()]
        dropRMol = Chem.FragmentOnBonds(self.mol, bondIndices=breakBondList, addDummies=False)
        doubleBondAtom1 = dropRMol.GetAtomWithIdx(doubleBondAtom1Idx)
        doubleBondAtom2 = dropRMol.GetAtomWithIdx(doubleBondAtom2Idx)
        if doubleBondAtom2.IsInRing() and doubleBondAtom1.IsInRing():
            self.derivativeMolDict[self.mol] = {'aziridineOnRingAtomIdxDict': {
                doubleBond.GetIdx(): [NAtomIdx, CH3AtomIdx,
                                      doubleBondAtom1Idx,
                                      doubleBondAtom2Idx]},
                'aziridineOnChainAtomIdxDict': {},
                'chiral': 0, 'rotonationChiral': 0
            }
        else:
            self.derivativeMolDict[self.mol] = {'aziridineOnRingAtomIdxDict': {},
                                                'aziridineOnChainAtomIdxDict': {
                                                    doubleBond.GetIdx(): [NAtomIdx, CH3AtomIdx,
                                                                          doubleBondAtom1Idx,
                                                                          doubleBondAtom2Idx]},
                                                'chiral': 0, 'rotonationChiral': 0
                                                }

        self.fragDerivation(oxhydrylBondBreakMaxNum=OHMaxBreak)
        self.calBreakCharacteristicMass(lowMsDectection=lowMsDetection, OHMaxBreak=OHMaxBreak)

    def derivativeMolAllPossible(self, derivativeNumber: Tuple[int, int] = (1, 1), sameCCW: int = 0,
                                 chargeNumber: Tuple[int, int] = (1, 1),
                                 derivativeType: int = 1, chiral: int = 0, protonationChiral: int = 0,
                                 useChiral: bool = True
                                 ):
        """
            Derivative ST molecule on all possible reaction site.DerivativeType=1 is Ts-Me, DerivativeType=2 is TS-H.
        :param useChiral: If True ,protonationChiral and chiral will be
        :param chargeNumber: Charge Number range. (min Number,max Number)
        :param sameCCW: 0 diff CCW,1 same CCW
        :param protonationChiral: protonation chiral
        :param chiral: c=c's c atom chiral
        :param derivativeType:1 TS-Me,2 TS-H
        :param derivativeNumber: Derivation Number range. (min Number,max Number)
        :return: A dict. The key of the dictionary is derivative molecule and the values is aziridines' N atom index list
        """
        if self.doubleBondNumber == 0:
            print('No reation site')
            return
        self._combinationsReactionAndChargeBondSite(derivativeNumber=derivativeNumber,
                                                    chargeNumber=chargeNumber)
        if derivativeType == 2:
            protonationChiral = -1
        if derivativeNumber[1] > self.doubleBondNumber:
            raise Exception(f'Derivation number max must less than double bond number {self.doubleBondNumber}')
        if chargeNumber[1] > derivativeNumber[0]:
            raise Exception(f'Molecule charge number max must less than derivative number min {derivativeNumber}')
        for reactionSiteTuple in self.reactionChargeSiteDict.keys():
            molTemp = copy.deepcopy(self.RWMol)
            aziridineOnRingAtomIdxDict = {}
            aziridineOnChainAtomIdxDict = {}
            for reactionBondIdx in reactionSiteTuple:
                (atom1, atom2) = (molTemp.GetBondWithIdx(reactionBondIdx).GetBeginAtomIdx(),
                                  molTemp.GetBondWithIdx(reactionBondIdx).GetEndAtomIdx())
                if derivativeType == 2:
                    aziridineRingAtomIdxList, afterDerivationMol = self._addAziridinesH(molTemp, sameCCW=sameCCW,
                                                                                        bondIdx=reactionBondIdx,
                                                                                        atomIdx=(atom1, atom2),
                                                                                        chiral=chiral,
                                                                                        useChiral=useChiral)
                if derivativeType == 1:
                    aziridineRingAtomIdxList, afterDerivationMol = self._addAziridinesMe(molTemp, sameCCW=sameCCW,
                                                                                         bondIdx=reactionBondIdx,
                                                                                         atomIdx=(atom1, atom2),
                                                                                         chiral=chiral,
                                                                                         protonationChiral=protonationChiral,
                                                                                         useChiral=useChiral)
                if reactionBondIdx in self.doubleBondBrokenIdx:
                    aziridineOnChainAtomIdxDict[reactionBondIdx] = aziridineRingAtomIdxList
                else:
                    aziridineOnRingAtomIdxDict[reactionBondIdx] = aziridineRingAtomIdxList
                aziridineAtomIdxList_Dict = {k: v for d in [aziridineOnChainAtomIdxDict,
                                                            aziridineOnRingAtomIdxDict] for k, v in d.items()}
            for chargeSiteList in self.reactionChargeSiteDict[reactionSiteTuple]:
                mol = copy.deepcopy(afterDerivationMol)
                for chargeSite in chargeSiteList:
                    mol.GetAtomWithIdx(aziridineAtomIdxList_Dict[chargeSite][0]).SetFormalCharge(1)
                mol.UpdatePropertyCache()
                # noinspection PyTypeChecker
                self.derivativeMolDict[mol] = dict(aziridineOnRingAtomIdxDict=aziridineOnRingAtomIdxDict,
                                                   aziridineOnChainAtomIdxDict=aziridineOnChainAtomIdxDict,
                                                   chiral=chiral, protonationChiral=protonationChiral)
        return self.derivativeMolDict

    def fragDerivation(self, oxhydrylBondBreakMaxNum: int = 1):
        """
            Single derivative and single charge can use frag derivative.
        """
        if not all(self.reactionSiteList) == 1:
            print('Derivation 1 and charge 1 need')
            return
        # ---------break in 2----------
        for preDerivationMol in self.derivativeMolDict.keys():
            self.afterDerivationBreakMolDict[preDerivationMol] = \
                self.fragAziridineOnDoubleBondFragIn2(preDerivationMol, self.derivativeMolDict[preDerivationMol])
        # --------drop ring----------
        tempDict = {}
        for x in self.derivativeMolDict.keys():
            tempDict[x] = []
        for preDerivationMol, afterDropChainMols in self.afterDerivationBreakMolDict.items():
            tempMols = []
            molInfo = self.derivativeMolDict[preDerivationMol]
            tempMols += self.fragAziridineOnRingDropRing(afterDropChainMols=afterDropChainMols, molInfo=molInfo)
            self.afterDerivationBreakMolDict[preDerivationMol] = tempMols
        # -----------frag ester bond------------
        tempDict = {}
        for x in self.derivativeMolDict.keys():
            tempDict[x] = []
        for preDerivationMol, afterMols in self.afterDerivationBreakMolDict.items():
            tempMols = []
            molInfo = self.derivativeMolDict[preDerivationMol]
            tempMols += self.fragEsterBond(afterDropChainRingMols=afterMols, molInfo=molInfo)
            self.afterDerivationBreakMolDict[preDerivationMol] = tempMols

    def fragEsterBond(self, afterDropChainRingMols: Mol, molInfo: Mol):
        afterMol = afterDropChainRingMols
        esterBondSiteList = self.CEBond
        esterBondListList = [-1, esterBondSiteList]
        afterFragEsterBondMolList = []
        if len(molInfo['aziridineOnRingAtomIdxDict']) > 0:
            onRing = True
        else:
            onRing = False
        for mol in afterMol:
            for esterBond in esterBondListList:
                if esterBond == {}:
                    continue
                tempMol = copy.deepcopy(mol)
                tempMol = self._fragEster(tempMol, esterBond, onRing)
                afterFragEsterBondMolList.append(tempMol)
        # for mol in afterFragEsterBondMolList:
        #     plot.showMol(mol)
        return afterFragEsterBondMolList

    def _fragEster(self, mol: Mol, afterBreakIn2MolsInfo: dict, onRing: bool):
        if afterBreakIn2MolsInfo == -1:
            return mol
        esterBondIdx = list(afterBreakIn2MolsInfo.keys())[0]
        esterCIdx = afterBreakIn2MolsInfo[esterBondIdx][0]
        esterOIdx = afterBreakIn2MolsInfo[esterBondIdx][1]
        esterBondIdx = [mol.GetBondBetweenAtoms(esterCIdx, esterOIdx).GetIdx()]
        fragMol = Chem.FragmentOnBonds(mol, bondIndices=esterBondIdx, addDummies=False)
        if onRing:
            oxBondCAtom = fragMol.GetAtomWithIdx(esterCIdx)
            self.minus1H(oxBondCAtom)
            bonds_ = oxBondCAtom.GetBonds()
            bonds = []
            for bond in bonds_:
                if bond.GetOtherAtom(oxBondCAtom).GetAtomicNum() == 6:
                    bonds.append(bond)
            for bond in bonds:
                if bond.GetBondType() != Chem.BondType.DOUBLE and \
                        self.otherAtomWithHsMoreThan1(bond, fragMol.GetAtomWithIdx(esterCIdx)):
                    otherC = bond.GetOtherAtom(oxBondCAtom)
                    bond.SetBondType(Chem.BondType.DOUBLE)
                    self.minus1H(otherC)
                    fragMol.UpdatePropertyCache()
                    break
                else:
                    pass
        if not onRing:
            oxBondCAtom = fragMol.GetAtomWithIdx(esterCIdx)
            self.minus1H(oxBondCAtom)
            oxBondCAtom.SetFormalCharge(1)
        return fragMol

    def fragOxhydrylBond(self, afterDropChainRingMols: Mol, oxhydrylBondBreakMaxNum: int = 1):
        """
            More than 1 frag has many problems.Such as complex electron transfer and rearrangement.Program possibly error
            due to incorrect valence number of C atoms. We can calculate ST mol’s frag ions mz with OH more than 0 using
            minus process in calBreakCharacteristicMass function. When want to get RDkit mol object only frag 1 OH Bond,
            using this function with oxhydrylBondBreakMaxNum = 1.This usage is always to show ST mol.
        :param afterDropChainRingMols:
        :param oxhydrylBondBreakMaxNum: How many times does ST mol frag.
        :return:
        """
        afterDropChainRingMols = afterDropChainRingMols
        oxhydrylBondSiteList = self.OHBond
        oxhydrylBondListList = []
        afterDropOxhydrylBondMolList = []
        for fragTimes in range(0, oxhydrylBondBreakMaxNum + 1):
            # 不要0次组合，直接返回原mol 排列组合可能的碎裂情况，一个双键链位点两种 只排列组合已经反应上的链的位点 比如两个链氮杂环 就有 [[a],[b],[a,b]]
            # 碎一次，碎一次，碎两次的三种情况，每次都会出现两种情况。能到 2+2+4种结果 当然会有重复
            for x in itertools.combinations(oxhydrylBondSiteList, fragTimes):
                if len(x) == 0:
                    oxhydrylBondListList.append([-1])  # -1 代表返回原值 不碎裂
                elif len(x) == 1:
                    oxhydrylBondListList.append([x[0]])
                elif len(x) > 1:
                    oxhydrylBondListList.append([y for y in x])

        for afterMol in afterDropChainRingMols:
            for oxhydrylBondList in oxhydrylBondListList:
                tempMol = copy.deepcopy(afterMol)
                for oxhydrylBond in oxhydrylBondList:
                    tempMol = self._fragOx(tempMol, oxhydrylBond)
                afterDropOxhydrylBondMolList.append(tempMol)
        return afterDropOxhydrylBondMolList

    def fragAziridineOnRingDropRing(self, afterDropChainMols: Mol, molInfo: dict):
        """
            返回去掉氮杂环的mol一定要返回一个没掉的，也就是无论传入什么，都要返回一个原值，代表为碎裂的mol也便和其他碎裂规则函数来生成全情况。
            每种衍生后分子的所有断氮杂环在链上情况。
        :param afterDropChainMols:
        :param molInfo:
        :return:
        """
        afterDropChainMols = afterDropChainMols
        doubleBondDropRSiteList = molInfo['aziridineOnRingAtomIdxDict']
        fragDropRSiteListList = []
        afterDropRingList = []
        # 排列组合
        for fragTimes in range(0, len(doubleBondDropRSiteList) + 1):
            # 不要0次组合，直接返回原mol 排列组合可能的碎裂情况，一个双键链位点两种 只排列组合已经反应上的链的位点 比如两个链氮杂环 就有 [[a],[b],[a,b]]
            # 碎一次，碎一次，碎两次的三种情况，每次都会出现两种情况。能到 2+2+4种结果 当然会有重复
            for x in itertools.combinations(doubleBondDropRSiteList, fragTimes):
                if len(x) == 0:
                    fragDropRSiteListList.append([-1])  # -1 代表返回原值 不碎裂
                elif len(x) == 1:
                    fragDropRSiteListList.append([x[0]])
                elif len(x) > 1:
                    fragDropRSiteListList.append([y for y in x])
        # 根据组合结果 碎键
        for afterDropChainMol in afterDropChainMols:
            for fragIn2SiteList in fragDropRSiteListList:
                tempMol = copy.deepcopy(afterDropChainMol)
                for fragIn2Site in fragIn2SiteList:
                    tempMol = self._fragR(tempMol, molInfo, fragIn2Site)
                afterDropRingList.append(tempMol)
        return afterDropRingList

    @staticmethod
    def _fragR(mol: Mol, afterBreakIn2MolsInfo: dict, dropRingSite):
        if dropRingSite == -1:
            return mol
        molInfo = afterBreakIn2MolsInfo['aziridineOnRingAtomIdxDict']
        atomIdxList = molInfo[dropRingSite]
        # newShowMol(mol, autoclose=False, showAtom=True)
        breakBondIdxList = [mol.GetBondBetweenAtoms(atomIdxList[0], atomIdxList[2]).GetIdx(),
                            mol.GetBondBetweenAtoms(atomIdxList[0], atomIdxList[3]).GetIdx()]

        dropRMol = Chem.FragmentOnBonds(mol, bondIndices=breakBondIdxList, addDummies=False)
        # newShowMol(dropRMol, autoclose=False, showAtom=True)
        dropRMol.GetBondBetweenAtoms(atomIdxList[2], atomIdxList[3]).SetBondType(Chem.BondType.DOUBLE)
        # newShowMol(dropRMol,autoclose=False,showAtom=True)
        dropRMol.UpdatePropertyCache()
        for x in [atomIdxList[2], atomIdxList[3]]:
            if dropRMol.GetAtomWithIdx(x).GetNumImplicitHs() >= 1:
                dropRMol.GetAtomWithIdx(x).SetFormalCharge(1)
                break
        dropRMol.UpdatePropertyCache()
        return dropRMol

    @staticmethod
    def _addAziridinesH(mol: Mol, atomIdx: Union[Tuple[int, int], type(None)], sameCCW=0,
                        bondIdx: Union[int, type(None)] = None, chiral: int = 0, useChiral: bool = True):
        """
            Ts-H derivative reactions. Need C atoms of double bond index or double bond index.
        :param mol: ST molecule need to derivative
        :param atomIdx:Two C atoms index of double bond
        :param bondIdx:Double bond index
        :param  charge:The molecule charge
        :return:N Atom Index, Rdkit RWMol class of derivatived molecule
        :param chiral:0,1 represent different chiral
        """
        if not bondIdx:
            bondIdx = mol.GetBondBetweenAtoms(atomIdx[0], atomIdx[1]).GetIdx()
        if not atomIdx:
            atomIdx[0] = mol.GetBondWithIdx(bondIdx).GetStartAtom()
            atomIdx[1] = mol.GetBondWithIdx(bondIdx).GetEndAtom()
        mw = Chem.RWMol(mol)
        # 设置手性
        if not useChiral:
            pass
        else:
            if sameCCW == 0:
                if chiral == 0:
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif chiral == 1:
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif sameCCW == 1:
                if chiral == 0:
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif chiral == 1:
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
        mw.GetBondWithIdx(bondIdx).SetBondType(Chem.BondType.SINGLE)
        NAtomIndex = mw.AddAtom(Chem.Atom(7))
        mw.AddBond(atomIdx[0], NAtomIndex, Chem.BondType.SINGLE)
        mw.AddBond(atomIdx[1], NAtomIndex, Chem.BondType.SINGLE)
        Chem.SanitizeMol(mw)
        return [NAtomIndex, atomIdx[0], atomIdx[1]], mw.GetMol()

    @staticmethod
    def _addAziridinesMe(mol: Mol, atomIdx: Union[Tuple[int, int], type(None)], sameCCW=0,
                         bondIdx: Union[int, type(None)] = None,
                         charge: int = 1, chiral: int = 0, protonationChiral: int = 0, useChiral: bool = True):
        """
            Ts-H derivative reactions. Need C atoms of double bond index or double bond index.
        :param mol: ST molecule need to derivative
        :param atomIdx:Two C atoms of double bond
        :param bondIdx:Double bond index
        :param charge:The molecule charge
        :return:Rdkit RWMol class of derivatived molecule
        :param charge:The molecule charge
        :param chiral:0,1 represent different chiral
        :param protonationChiral:0,1 represent different protonation chiral
        """
        mw = mol
        if not bondIdx:
            bondIdx = mw.GetBondBetweenAtoms(atomIdx[0], atomIdx[1]).GetIdx()
        if not atomIdx:
            atomIdx[0] = mw.GetBondWithIdx(bondIdx).GetStartAtom()
            atomIdx[1] = mw.GetBondWithIdx(bondIdx).GetEndAtom()
        # 双键改单键
        mw.GetBondWithIdx(bondIdx).SetBondType(Chem.BondType.SINGLE)
        # 设置手性
        if not useChiral:
            pass
        else:
            if sameCCW == 0:
                if chiral == 0:
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif chiral == 1:
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif sameCCW == 1:
                if chiral == 0:
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif chiral == 1:
                    mw.GetAtomWithIdx(atomIdx[1]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
                    mw.GetAtomWithIdx(atomIdx[0]).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
        # 增加 N-ME
        NAtomIndex = mw.AddAtom(Chem.Atom(7))
        CAtomIndex = mw.AddAtom(Chem.Atom(6))
        if not useChiral:
            pass
        else:
            if protonationChiral == 0:
                mw.GetAtomWithIdx(NAtomIndex).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif protonationChiral == 1:
                mw.GetAtomWithIdx(NAtomIndex).SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        mw.AddBond(CAtomIndex, NAtomIndex, Chem.BondType.SINGLE)
        mw.AddBond(atomIdx[0], NAtomIndex, Chem.BondType.SINGLE)
        mw.AddBond(atomIdx[1], NAtomIndex, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(mw)
        except:
            print('wrong', Chem.MolToSmiles(mw.GetMol()))
        return [NAtomIndex, CAtomIndex, atomIdx[0], atomIdx[1]], mw.GetMol()

    def generate3DFile(self, molName: str, filePath: str):
        self.derivativeMolAllPossible(chiral=0, protonationChiral=0)
        self.derivativeMolAllPossible(chiral=1, protonationChiral=1)
        self.derivativeMolAllPossible(chiral=0, protonationChiral=1)
        self.derivativeMolAllPossible(chiral=1, protonationChiral=0)
        if len(self.derivativeMolDict) == 0:
            print(f'None mol generate')
        else:
            smileDict = {}
            for i, atomSite in enumerate(self.derivativeMolDict):
                for molDict in self.derivativeMolDict[atomSite]:
                    molNameStr = ''
                    mol = next(iter(molDict))
                    regex = '%s_%s_%s'
                    molNameStr = regex % (str(i), str(molDict[mol][1]), str(molDict[mol][2]))
                    filePathMol = "./" + molNameStr + '.mol'
                    filePathPng = "./" + molNameStr + '.png'
                    Chem.MolToMolFile(mol, filename=filePathMol)
                    import pandas as pd
                    smileDict[molNameStr] = Chem.MolToSmiles(mol)
                smilePd = pd.DataFrame.from_dict(smileDict, orient='index')
                try:
                    smilePd.to_csv('./TS-H_Smiles_No_CCW.csv')
                except PermissionError:
                    smilePd.to_csv('./1_1.csv')

    def generateDerivationSmiles(self, molName: str, sameCCW: int = 0, derivativeNumber: Tuple[int, int] = (1, 1),
                                 chargeNumber: Tuple[int] = (1, 1), chiral: List[int] = None, useChiral: bool = True,
                                 protonationChiral: List[int] = None, derivativeType: int = 1):
        """

        :param useChiral:
        :param sameCCW:
        :param molName:
        :param derivativeNumber: add derivative mol number
        :param chargeNumber: Tuple[int] = (1, 1),
        :param derivativeType: 1 TS-Me,2 TS-H
        :param chiral:0,1 represent different chiral
        :param protonationChiral:0,1 represent different protonation chiral
        :return:
        """
        if not chiral:
            chiral = [0]
        if not protonationChiral:
            protonationChiral = [0]
        for x in chiral:
            for y in protonationChiral:
                # print(x, y)
                self.derivativeMolAllPossible(derivativeNumber=derivativeNumber, sameCCW=sameCCW,
                                              chargeNumber=chargeNumber, useChiral=useChiral,
                                              derivativeType=derivativeType, chiral=x,
                                              protonationChiral=y)
        if len(self.derivativeMolDict) == 0:
            print(f'None mol generate')
        else:
            derivativeAllMolDict = {}
            for rwMol in self.derivativeMolDict:
                molInfo = self.derivativeMolDict[rwMol]
                ringDict = molInfo['aziridineOnRingAtomIdxDict']
                chainDict = molInfo['aziridineOnChainAtomIdxDict']
                if ringDict:
                    atomSiteStr = str(ringDict[next(iter(ringDict))][-2]) + '-' + str(
                        ringDict[next(iter(ringDict))][-1])
                    ringFLag = 'ring'
                if chainDict:
                    atomSiteStr = str(chainDict[next(iter(chainDict))][-2]) + '-' + str(
                        chainDict[next(iter(chainDict))][-1])
                    ringFLag = 'chain'
                chiralStr = str(molInfo['chiral'])
                protonationChiralStr = str(molInfo['protonationChiral'])
                molNameStr = '%s_%s_%s_%s_%s' % (molName, atomSiteStr, chiralStr, protonationChiralStr, ringFLag)
                derivativeAllMolDict[molNameStr] = Chem.MolToSmiles(rwMol)
            return derivativeAllMolDict

    def fragAziridineOnDoubleBondFragIn2(self, mol, molInfo):
        derivativeRwMol = mol
        derivativeRwMolInfo = molInfo
        doubleBondBreakSiteIdx = derivativeRwMolInfo['aziridineOnChainAtomIdxDict']
        fragIn2SiteListList = []
        fragIn2Mols = [derivativeRwMol]
        for fragTimes in range(1, len(doubleBondBreakSiteIdx) + 1):
            # 不要0次组合，直接返回原mol 排列组合可能的碎裂情况，一个双键链位点两种 只排列组合已经反应上的链的位点 比如两个链氮杂环 就有 [[a],[b],[a,b]]
            # 碎一次，碎一次，碎两次的三种情况，每次都会出现两种情况。能到 2+2+4种结果 当然会有重复
            for x in itertools.combinations(doubleBondBreakSiteIdx, fragTimes):
                if len(x) == 1:
                    fragIn2SiteListList.append([x[0]])
                else:
                    fragIn2SiteListList.append([y for y in x])
        for fragIn2SiteList in fragIn2SiteListList:
            tempMol = copy.deepcopy(derivativeRwMol)
            for fragIn2Site in fragIn2SiteList:
                tempMol = self._fragin2(tempMol, derivativeRwMolInfo, fragIn2Site)
            fragIn2Mols += tempMol
        return fragIn2Mols

    def _fragOx(self, mol, dropOxSite):
        if dropOxSite == -1:
            return mol
        oxBondCAtomIdx = self.OHBond[dropOxSite][1]
        oxBondIdx = mol.GetBondBetweenAtoms(self.OHBond[dropOxSite][0], self.OHBond[dropOxSite][1]).GetIdx()
        dropOxMol = Chem.FragmentOnBonds(mol, bondIndices=[oxBondIdx], addDummies=False)
        dropOxMol.UpdatePropertyCache()
        oxBondCAtom = dropOxMol.GetAtomWithIdx(oxBondCAtomIdx)
        self.minus1H(oxBondCAtom)
        bonds_ = dropOxMol.GetAtomWithIdx(oxBondCAtomIdx).GetBonds()
        bonds = []
        for bond in bonds_:
            if bond.GetOtherAtom(oxBondCAtom).GetAtomicNum() == 6:
                bonds.append(bond)
        for bond in bonds:
            if bond.GetBondType() != Chem.BondType.DOUBLE and self.otherAtomWithHsMoreThan1(bond,
                                                                                            dropOxMol.GetAtomWithIdx(
                                                                                                oxBondCAtomIdx)):
                otherC = bond.GetOtherAtom(oxBondCAtom)
                # print(otherC.GetIdx(), otherC.GetNumExplicitHs(), otherC.GetNumImplicitHs(), otherC.GetTotalValence())
                bond.SetBondType(Chem.BondType.DOUBLE)
                self.minus1H(otherC)
                # print(oxBondCAtom.GetIdx(), oxBondCAtom.GetNumExplicitHs(), oxBondCAtom.GetNumImplicitHs(),
                #       oxBondCAtom.GetTotalValence())
                dropOxMol.UpdatePropertyCache()
                break
            else:
                pass
        return dropOxMol

    @staticmethod
    def minus1H(atom):
        total = atom.GetTotalNumHs()
        if total >= 1:
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(total - 1)

    @staticmethod
    def add1H(atom):
        total = atom.GetTotalNumHs()
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total + 1)

    @staticmethod
    def otherAtomWithHsMoreThan1(bond, atom):
        otherAtom = bond.GetOtherAtom(atom)
        if otherAtom.GetNumExplicitHs() + otherAtom.GetNumImplicitHs() >= 1:
            return True
        else:
            return False

    @staticmethod
    def bondWithExplicitHsMoreThan1(bond):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if atom1.GetNumExplicitHs() + atom1.GetNumImplicitHs() >= 1 or \
                atom2.GetNumExplicitHs() + atom2.GetNumImplicitHs() >= 1:
            return True
        else:
            return False

    def _fragin2(self, mols, molInfo, needBreakSite):
        if type(mols) != list:
            mols = [mols]
        fragmols = []
        for mol in mols:
            breakBondInfoListList = []
            [NAtomIdx, CH3AtomIdx, doubleBondAtom1, doubleBondAtom2] = molInfo['aziridineOnChainAtomIdxDict'][
                needBreakSite]
            NC1BondIdx = mol.GetBondBetweenAtoms(NAtomIdx, doubleBondAtom1).GetIdx()
            NC2BondIdx = mol.GetBondBetweenAtoms(NAtomIdx, doubleBondAtom2).GetIdx()
            willBeDoubleBondIdx = mol.GetBondBetweenAtoms(doubleBondAtom1, doubleBondAtom2).GetIdx()
            breakBondInfoListList.append(
                [([NC1BondIdx, willBeDoubleBondIdx], (NAtomIdx, doubleBondAtom2), doubleBondAtom1),
                 ([NC2BondIdx, willBeDoubleBondIdx], (NAtomIdx, doubleBondAtom1), doubleBondAtom2)])
            # 碎一个位点 的两种情况的需要的键信息和原子信息，接下来每次操作改了删除了键，动了序号。但是没有动原子序号，所以接下来某些操作要使用getbond重新获得键信息
            for breakBondInfoList in breakBondInfoListList:
                for breakBondInfo in breakBondInfoList:
                    fragmol = Chem.FragmentOnBonds(mol, bondIndices=breakBondInfo[0], addDummies=False)
                    fragmol.GetBondBetweenAtoms(breakBondInfo[1][0], breakBondInfo[1][1]).SetBondType(
                        Chem.BondType.DOUBLE)
                    self.minus1H(fragmol.GetAtomWithIdx(breakBondInfo[1][1]))
                    fragmol.GetAtomWithIdx(breakBondInfo[1][1]).UpdatePropertyCache()
                    fragmol.GetAtomWithIdx(NAtomIdx).SetNumExplicitHs(1)
                    fragmol.GetAtomWithIdx(NAtomIdx).UpdatePropertyCache()
                    c2cBond = fragmol.GetAtomWithIdx(breakBondInfo[2]).GetBonds()
                    if len(c2cBond) > 0:
                        for bond in c2cBond:
                            if self.otherAtomWithHsMoreThan1(bond, mol.GetAtomWithIdx(breakBondInfo[2])):
                                c2cBondOtherAtom = bond.GetOtherAtom(fragmol.GetAtomWithIdx(breakBondInfo[2]))
                                bond.SetBondType(Chem.BondType.DOUBLE)
                                bond.SetBondDir(Chem.rdchem.BondDir.NONE)
                                # self.deleteStereo(c2cBondOtherAtom)
                                self.minus1H(c2cBondOtherAtom)
                                explicitHs = c2cBondOtherAtom.GetNumExplicitHs()
                                if explicitHs >= 1:
                                    c2cBondOtherAtom.SetNumExplicitHs(explicitHs - 1)
                                fragmol.UpdatePropertyCache()
                                break
                    fragmols.append(fragmol)
        return fragmols

    def calBreakCharacteristicMass(self, lowMsDectection: float = 100, OHMaxBreak: int = 1):
        """
        This function calculate break mass, break in chain use emass, break in ring use minus neutral loss

        :param lowMsDectection: low mass range of mass spectrometer
        :param OHMaxBreak: neutral mass loss of H2O number
        :return:
        """
        emassCalculator = EMass()
        for mol in self.derivativeMolDict.keys():
            if len(self.derivativeMolDict[mol]['aziridineOnChainAtomIdxDict']) != 0:
                isChain = True
            else:
                isChain = False
            derivativeMol = mol
            formula = CalcMolFormula(derivativeMol).split('+')[0]
            noBreakMolMass = round(emassCalculator.calculate(formula=formula, charge=1, limit=0.001)[0].mass, 6)
            self.noBreakMolMass = noBreakMolMass
            if OHMaxBreak <= self.OHBondNumber:
                OHMaxBreak = OHMaxBreak
            else:
                OHMaxBreak = self.OHBondNumber
            # --------------------先提取衍生链上双键的离子-------------------
            if len(self.CEBond) > 0:
                for mol in self.afterDerivationBreakMolDict[derivativeMol]:
                    smiles = Chem.MolToSmiles(mol)
                    splitSmilesList = smiles.split('.')
                    for smile in splitSmilesList:
                        if '+' in smile and len(splitSmilesList) > 1:
                            withChargeMol = Chem.MolFromSmiles(smile)
                            formula = CalcMolFormula(withChargeMol)
                            formula = formula.split('+')[0]
                            fragEmassShift = round(
                                emassCalculator.calculate(formula=formula, charge=1, limit=0.01)[0].mass,
                                6)
                            if fragEmassShift != noBreakMolMass:
                                # print(fragEmassShift)
                                self.characteristicMassList.append(fragEmassShift)
                self.characteristicMassList = [round(x, 5) for x in self.characteristicMassList if x > lowMsDectection]
                self.characteristicMassList = list(set(self.characteristicMassList))
                self.characteristicMassList.sort()
                if isChain:
                    self.diagIonsInChain = self.characteristicMassList
                else:
                    self.diagIonsInRing = self.characteristicMassList
                return
            if len(self.CEBond) <= 1:
                if isChain:
                    for mol in self.afterDerivationBreakMolDict[derivativeMol]:
                        mol = Chem.RemoveAllHs(mol)
                        smiles = Chem.MolToSmiles(mol)
                        splitSmilesList = smiles.split('.')
                        for smile in splitSmilesList:
                            if '+' in smile and len(splitSmilesList) > 1:
                                withChargeMol = Chem.MolFromSmiles(smile)
                                formula = CalcMolFormula(withChargeMol)
                                OHTuple = withChargeMol.GetSubstructMatches(
                                    Chem.MolFromSmarts(radicalGroupSmart.anyOxhydrylBondSmart))
                                if OHMaxBreak <= len(OHTuple):
                                    OHMaxBreakInBreak = OHMaxBreak
                                else:
                                    OHMaxBreakInBreak = len(OHTuple)
                                OHShiftInBreakIons = np.arange(0, OHMaxBreakInBreak + 1, 1) * 18.01056
                                formula = formula.split('+')[0]
                                fragEmassShift = round(
                                    emassCalculator.calculate(formula=formula, charge=1, limit=0.01)[0].mass,
                                    6)
                                if fragEmassShift != noBreakMolMass:
                                    result = round(fragEmassShift, 6) - OHShiftInBreakIons
                                    if type(result) is not float:
                                        for x in result:
                                            if float(x) > lowMsDectection:
                                                self.characteristicMassList.append(x)
                                    else:
                                        if float(result) > lowMsDectection:
                                            self.characteristicMassList.append(result)
                    self.characteristicMassList = [round(x, 5) for x in self.characteristicMassList if x > lowMsDectection]
                    self.diagIonsInChain = self.characteristicMassList
                else:
                    if OHMaxBreak <= len(self.OHBond):
                        OHMaxBreakInBreak = OHMaxBreak
                    else:
                        OHMaxBreakInBreak = len(self.OHBond)
                    OHShiftOH = np.arange(0, OHMaxBreakInBreak + 1, 1) * 18.01056
                    formula = self.formula.split('+')[0]
                    mz = round(emassCalculator.calculate(formula=formula, charge=1, limit=0.01)[0].mass,6)
                    self.characteristicMassList = mz - OHShiftOH - 31.0422
                    self.diagIonsInRing = self.characteristicMassList

    def findDoubleBondLocationInSTRings(self, mol: Mol = None):
        """
            Find the double bond position of the double bond in the ring.Bond index definite in ringIndex.png
            Has problem : Index order is not same as academic standards.
        """
        if mol:
            mol = mol
        else:
            mol = self.mol
        if self.isStandard:
            locationList = []
            for key in self._locationTemplateMolDict.keys():
                res = rdFMCS.FindMCS([self._locationTemplateMolDict[key], mol],
                                     bondCompare=rdFMCS.BondCompare.CompareAny)
                if res.smartsString.count(',') != res.smartsString.count('='):
                    locationList.append(int(key))
                    self.doubleBondLocInSTRingList = locationList
        return self.doubleBondLocInSTRingList

    def deleteStereo(self, atom):
        for bond in atom.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)