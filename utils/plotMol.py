from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem, Draw
from typing import Union, List, Tuple
import math
import os
import tkinter
from PIL import ImageTk, Image
from io import BytesIO
from rdkit.Chem import AllChem as aChem
from rdkit import Chem
import threading
import time


def showMol(mol: Union[Mol, None], size: Tuple[int, int] = (500, 500), title: str = 'RDKit Molecule',
            rotateDegree: float = 0, showAtom: bool = False, showBond: bool = False,
            stayInFront=True, showStereoAnnotation: bool = False, autoclose:bool = True,
            closeTime: int = 2, useTemp: Union[bool, str] = False, clearMap: str = True, *args, **kwargs):
    """     
        Simply show molecules
    :param autoclose:
    :param savePath:
    :param clearMap: clear map num default True
    :param showBond: show bond idx num
    :param useTemp: use a template to show 
    :param closeTime: auto close ticker window time unit second
    :param rotateDegree: pic rotate degree
    :param showStereoAnnotation: show atom CIP(R/S) in pic
    :param stayInFront: window show in front
    :param showAtom: show atom index in pic
    :param title: pic title
    :param mol: rdkit molecule object
    :param size: pic size (ixx,ixy)
    :return:
    """
    if type(useTemp) is str:
        try:
            tempMol = Chem.MolFromSmarts(useTemp)
            aChem.Compute2DCoords(tempMol)
            aChem.GenerateDepictionMatching2DStructure(mol, tempMol)
        except Exception:
            print('Plot with temp failed')
    else:
        pass
    if clearMap:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    Chem.RemoveHs(mol)
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    d2d.drawOptions().addAtomIndices = showAtom
    d2d.drawOptions().addBondIndices = showBond
    d2d.drawOptions().addStereoAnnotation = showStereoAnnotation
    d2d.drawOptions().rotate = rotateDegree
    d2d.drawOptions().dummyIsotopeLabels = True
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

    def close_window(closeTime):
        tkRoot.update()
        time.sleep(closeTime)
        tkRoot.quit()
        tkRoot.destroy()

    if autoclose:
        threading.Thread(target=close_window(closeTime)).start()
    tkRoot.mainloop()
    return d2d


def showReaction(reaction: Union[Mol, None], size: Tuple[int, int] = (1000, 1000), title: str = 'RDKit Molecule',
                 rotateDegree: float = 0, showAtom: bool = False, stayInFront=True, showStereoAnnotation: bool = False,
                 autoclose: int = 1.5, *args, **kwargs):
    """
        Simply show molecules
    :param autoclose:
    :param reaction:
    :param rotateDegree: pic rotare degree
    :param showStereoAnnotation: show atom CIP(R/S) in pic
    :param stayInFront:window show in front
    :param showAtom: show atom index in pic
    :param title: pic title
    :param mol: rdkit molecule object
    :param size: pic size (ixx,ixy)
    :return:
    """
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    d2d.drawOptions().addAtomIndices = False
    d2d.DrawReaction(reaction, **kwargs)
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

    def close_window(autoclose):
        tkRoot.update()
        time.sleep(autoclose)
        tkRoot.quit()
        tkRoot.destroy()

    if autoclose:
        threading.Thread(target=close_window(autoclose)).start()
    tkRoot.mainloop()
    return d2d


def saveMol(savePath: str, mol: Union[Mol, None], size: Tuple[int, int] = (500, 500), title: str = 'RDKit Molecule',
            rotateDegree: float = 0, showAtom: bool = False, showStereoAnnotation: bool = False,
            *args, **kwargs):
    """

    :param title:
    :param showStereoAnnotation:
    :param size:
    :param rotateDegree:
    :param mol:
    :param text:
    :param showBond:
    :param showAtom:
    :param savePath:
    :param template:
    :param save:
    :return:
    """
    # tempMol = Chem.MolFromSmarts(
    #     '[#6]12~[#6]~[#6]~[#6]3~[#6]4~[#6]~[#6]~[#6]~[#6]~4~[#6]~[#6]~[#6]~3~[#6]~1~[#6]~[#6]~[#6]~[#6]~2')
    # aChem.Compute2DCoords(tempMol)
    # aChem.GenerateDepictionMatching2DStructure(mol, tempMol)
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    d2d.drawOptions().addAtomIndices = showAtom
    d2d.drawOptions().addStereoAnnotation = showStereoAnnotation
    d2d.drawOptions().rotate = rotateDegree
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg_text = d2d.GetDrawingText()
    with open(savePath, 'wb') as png_file:
        png_file.write(svg_text)
    png_file.close()
    return True


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
