from rdkit import Chem
from rdkit.Chem import AllChem
import myplot
from typing import List, Union
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

tempRingUnspecifiedSmarts = \
    '[#6]12~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~1~[#6]~[#6]~[#6]~[#6]~2'
tempRingSingleSmarts = \
    '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2'
tempRingUnspecifiedMol = Chem.MolFromSmarts(tempRingUnspecifiedSmarts)
tempRingSingleMol = Chem.MolFromSmarts(tempRingSingleSmarts)
tempRingLocSmartsDict = {
    0: "[#6]12=[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2",
    1: '[#6]12-[#6]=[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    2: '[#6]12-[#6]-[#6]=[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    3: '[#6]12-[#6]-[#6]-[#6]3=[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    4: '[#6]12-[#6]-[#6]-[#6]3-[#6]4=[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    5: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]=[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    6: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]=[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    7: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]=[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    8: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4=[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    9: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]=[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    10: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]=[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    11: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3=[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    12: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1=[#6]-[#6]-[#6]-[#6]-2',
    13: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]=[#6]-[#6]-[#6]-2',
    14: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]=[#6]-[#6]-2',
    15: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]=[#6]-2',
    16: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]=2',
    17: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]-3-[#6]=1-[#6]-[#6]-[#6]-[#6]-2',
    18: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]-4-[#6]-[#6]-[#6]=3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2',
    19: '[#6]12-[#6]-[#6]-[#6]3-[#6]4-[#6]-[#6]-[#6]-[#6]=4-[#6]-[#6]-[#6]-3-[#6]-1-[#6]-[#6]-[#6]-[#6]-2'
}
temRingLocMolDict = {}
for key in tempRingLocSmartsDict.keys():
    temRingLocMolDict[key] = Chem.MolFromSmarts(tempRingLocSmartsDict[key])
cholesterolSmiles = 'CC(CCC[C@H]([C@H]1CC[C@H]2[C@@H]3CC=C4C[C@H](CC[C@@]4([C@H]3CC[C@@]21C)C)O)C)C'
cholesterolMol = Chem.MolFromSmiles(cholesterolSmiles)
cholesterolMorgan = GetMorganFingerprintAsBitVect(cholesterolMol, 2)


class template:
    def __init__(self):
        self._locationTemplateMolDict:dict = {}

    def _generateTemplateSingleDoubleBondSTRings(self):  # 生成20个代表不同位子单键长的环模板
        """
            Generate 20 template ST rings using generateTemplateDoubleBondsSTRingsFromList(),every mol represents a
            location.
        """
        for i in range(0, 20):
            templateMol = self.generateTemplateDoubleBondsSTRingsFromList([i])
            self._locationTemplateMolDict[i] = templateMol
        self._templateRingsMol = self.generateTemplateDoubleBondsSTRingsFromList(None, save=False)
        myplot.showMol(self._templateRingsMol)

    @staticmethod
    def generateTemplateDoubleBondsSTRingsFromList(bondLocation: Union[List[int], type(None)], save=True) -> Mol:
        """
        Generate ST ring in the same direction with specified double bond position.Use
        GenerateDepictionMatching2DStructure() function to ensure that the two-dimensional coordinate direction of
        molecules is consistent. Returns a cholesterol molecular template with double bonds on the given location.

        NOTE: When bondLocation is None, return ST rings every bond is SINGLE.
        :param bondLocation: A location index list(int) or None. Need list [1] not int 1.If None, return a mol with
                            every bond is UNSPECIFIED
        :param save: If True, sameCCW a png with ring index figure
        :return: Chem Mol object. Return a cholesterol molecular template rdkit Mol object with double bonds on the
                given location
        """
        smiles = 'C21CCC3C4CCCC4CCC3C2CCCC1'
        smiles1 = 'C12CCCCC1CCCC2'
        templateMol = Chem.MolFromSmiles(smiles)
        templateMol1 = Chem.MolFromSmiles(smiles1)
        AllChem.Compute2DCoords(templateMol1)
        AllChem.GenerateDepictionMatching2DStructure(templateMol, templateMol1)  # using template with same direction
        # print(bondLocation)
        if save:
            myplot.showMolWithIdx(templateMol, showBond=True, showAtom=True, savePath='./ringIndex.png')
        if isinstance(bondLocation, int):
            bondLocation = [bondLocation]
        if isinstance(bondLocation, type(None)) or len(bondLocation) == 0:
            return templateMol
        if isinstance(bondLocation, list):
            if max(bondLocation) > 19 or min(bondLocation) < 0:
                raise Exception(f'BondLocation 0-19')
            else:
                for index in bondLocation:
                    templateMol.GetBondWithIdx(index).SetBondType(Chem.rdchem.BondType.DOUBLE)
                return templateMol
        else:
            raise Exception(f'Need List[int] or None Now is {type(bondLocation)}')

# if __name__ == '__main__':
#     tempRingUnspecifiedSmarts = str
#     templateRingLocationList = []
#     tem = template()
#     temRingLocMolDict = {}
#     temRingLocSmartsDict = {}
#     templateRingMol = tem.generateTemplateDoubleBondsSTRingsFromList(None, sameCCW=None)
#     for bond in templateRingMol.GetBonds():
#         print(bond.GetBondType())
#     tempRingUnspecifiedSmarts = Chem.MolToSmarts(templateRingMol)
#     mol = tem.generateTemplateDoubleBondsSTRingsFromList(2, sameCCW=None)
#     for i in range(0, 20):
#         temRingLocMolDict[i] = tem.generateTemplateDoubleBondsSTRingsFromList(i, sameCCW=None)
#     for key in temRingLocMolDict.keys():
#         temRingLocSmartsDict[key] = Chem.MolToSmarts(temRingLocMolDict[key])
#     print(temRingLocSmartsDict)
