import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem, Draw
from rdkit import Chem
from typing import Union, List, Tuple
import math
import os
from utils import utils
import tkinter
from PIL import ImageTk, Image
from io import BytesIO
import threading
import time


def newShowMol(mol: Union[Mol, None], size: Tuple[int, int] = (500, 500), title: str = 'RDKit Molecule',
               rotateDegree: float = 0, showAtom: bool = False, showBond: bool = False,
               stayInFront=True, showStereoAnnotation: bool = False,
               autoclose: int = 1.5, usetemp: bool = True, *args, **kwargs):
    """
        Simply show molecules
    :param usetemp: If use temp
    :param autoclose: if is a number ,thinker window will auto close in number seconds. If is none, thinker window need
                      be shutdown manual
    :param rotateDegree: pic rotare degree
    :param showStereoAnnotation: show atom CIP(R/S) in pic
    :param stayInFront:window show in front
    :param showAtom: show atom index in pic
    :param title: pic title
    :param mol: rdkit molecule object
    :param size: pic size (ixx,ixy)
    :return:
    """
    if usetemp:
        try:
            tempMol = Chem.MolFromSmarts('[#6]12~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~1~[#6]~[#6]~[#6]~[#6]~2')
            AllChem.Compute2DCoords(tempMol)
            AllChem.GenerateDepictionMatching2DStructure(mol, tempMol)
        except:
            pass
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    d2d.drawOptions().addAtomIndices = showAtom
    d2d.drawOptions().addBondIndices = showBond
    d2d.drawOptions().addStereoAnnotation = showStereoAnnotation
    d2d.drawOptions().rotate = rotateDegree
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    sio = BytesIO(d2d.GetDrawingText())
    img = Image.open(sio)

    tkRoot = tkinter.Tk()
    tkRoot.title(title)
    tkPI = ImageTk.PhotoImage(img)
    tkLabel = tkinter.Label(tkRoot, image=tkPI)
    tkLabel.place(x=0, y=0, width=img.size[0], height=img.size[1])
    tkRoot.geometry('%dx%d' % img.size)
    tkRoot.lift()
    if stayInFront:
        tkRoot.attributes('-topmost', True)

    def close_window(autoclose_):
        tkRoot.update()
        time.sleep(autoclose_)
        tkRoot.quit()
        tkRoot.destroy()

    if autoclose:
        threading.Thread(target=close_window(autoclose)).start()
    tkRoot.mainloop()
    return d2d


def showMol(mol: Union[Mol, None], size: Tuple[int, int] = (500, 500), title: str = 'RDKit Molecule'):
    """
        Simply draw molecules
    :param title:
    :param mol:
    :param size:
    :return:
    """
    if mol:
        Draw.ShowMol(mol, size=size, title=title)
        return 1
    else:
        return 0


def showMolWithIdx(mol: Mol, text: str = None, showBond: bool = False, showAtom: bool = False,
                   savePath: Union[None, str] = None, template: Mol = None,
                   save: bool = True):
    timeStr = utils.getTime()
    randomNum = np.random.randint(1000, dtype='l')
    if save:
        if isinstance(savePath, str):
            savePath = savePath
        else:
            savePath = f'./{timeStr}_{randomNum}.png'

    if type(mol) is str:
        mol = Chem.MolFromSmiles(mol)
        # showMol(mol)
        # showMol(template)
    if template:
        AllChem.GenerateDepictionMatching2DStructure(mol, template)
    d2d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
    d2d.drawOptions().addBondIndices = showBond
    d2d.drawOptions().addAtomIndices = showAtom
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    png_text = d2d.GetDrawingText()
    with open(savePath, 'wb') as png_file:
        png_file.write(png_text)
    png_file.close()
    return None


def templateGridMol(mols: Union[List[str], List[Mol]], names: List[str], template: Union[str, Mol],
                    savePath: str = None, saveName: str = None, show: bool = None) -> list:
    """
        Batch plotResult.py molecules in the input template direction.
    :param mols:
    :param names:
    :param template:
    :param savePath:
    :param saveName:
    :param show:
    """
    molList = []
    for mol in mols:
        if type(mol) is str:
            mol = Chem.MolFromSmiles(mol)
        # showMol(mol)
        # showMol(template)
        AllChem.GenerateDepictionMatching2DStructure(mol, template)
        molList.append(mol)
    molNumber = len(molList)
    figureNumber = math.ceil(molNumber / 49)
    for i in range(0, figureNumber):
        img = Draw.MolsToGridImage(
            molList[i * 49: (i + 1) * 49],  # mol对象
            legends=names[i * 49: (i + 1) * 49],
            molsPerRow=7,
            subImgSize=(1000, 1000)
        )
        if show:
            img.show()
        if savePath:
            fileName = saveName + '_' + str(i) + '.jpg'
            img.save(os.path.join(savePath, fileName))
