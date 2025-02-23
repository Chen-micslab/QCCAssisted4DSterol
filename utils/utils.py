from typing import List, Union
import os
import math
from rdkit import Chem
import spectral_similarity
import numpy as np


def listToStr(strList: Union[List[int], None], delimiter: str = ',') -> str:  # 将int列表转为以逗号为分割的字符串
    if strList:
        listLikeStr = delimiter.join([str(s) for s in strList])
    else:
        listLikeStr = 'None'
    return listLikeStr


def listLikeStrNumber(listLikeStr: Union[str, None], delimiter: str = ',') -> int:  # 返回逗号分割字符里的int个数
    if listLikeStr == 'None':
        return 0
    else:
        return len(listLikeStr.split(delimiter))


def strToList(listLikeStr: Union[str, None], delimiter: str = ',') -> list:  # 逗号分割的字符串
    if isinstance(listLikeStr, str):
        if listLikeStr == 'None':
            return []
        else:
            return [int(x) for x in listLikeStr.split(delimiter)]
    else:
        return []


def getTime():
    import time
    now = int(round(time.time() * 1000))
    timeTuple = time.localtime(now / 1000)
    year = timeTuple[0] - 2000
    mon = timeTuple[1]
    day = timeTuple[2]
    hour = timeTuple[3]
    minute = timeTuple[4]
    second = timeTuple[5]
    return '{:0>2d}{:0>2d}{:0>2d}-{:0>2d}{:0>2d}{:0>2d}'.format(year, mon, day, hour, minute, second)


def findsingleLine(lineChar: str, content: list):
    count = 0
    for line in content:
        if lineChar in line:
            return count
        else:
            count += 1


def createDict(tupleVals, val):
    tupNum = len(tupleVals)
    tempName = dictName = {}
    for index, tupleVal in enumerate(tupleVals):
        if index == tupNum - 1:
            dictName[tupleVal] = val
        else:
            if tupleVal not in dictName.keys():
                dictName[tupleVal] = {}
            else:
                dictName.update({tupleVal: {}})
        dictName = dictName[tupleVal]
    return tempName


def getfilepath(folder, kindofFile):
    """
        调用os得到当前目录的所有文件名字的列表
        :param filepath:
        :return: 所有文件名字的列表
    """
    filePathList = []
    fileNameList = os.listdir(folder)
    fileNamePopList = []
    i = 0
    for fileName in fileNameList:
        if os.path.splitext(fileName)[1] == kindofFile:
            # print(fileName)
            i += 1
            filePathList.append(os.path.join(folder, fileName))
        else:
            fileNamePopList.append(i)
            pass
    for idx in fileNamePopList:
        fileNameList.remove(fileNameList[idx])
    return filePathList, fileNameList


def calculate_k0(mz, ccs, charge=1):
    m = 28.00615
    t = 304.7527
    coeff = 18500 * charge * math.sqrt((mz * charge + m) / (mz * charge * m * t))
    k0 = ccs / coeff
    return k0


def calculate_CCS(mz, k0, charge=1):
    m = 28.00615
    t = 304.7527
    coeff = 18500 * charge * math.sqrt((mz * charge + m) / (mz * charge * m * t))
    ccs = k0 * coeff
    return ccs


def SmilesToSmarts(smiles: str):
    return Chem.MolToSmarts(Chem.MolFromSmiles(smiles))


def ms2Dot(ms2Library: np.array, ms2Query: np.array, ppm: float = 30.0, method: str = 'cosine'):
    """

    :param ppm:
    :param ms2Array1: ms2 library
    :param ms2Array2: ms2 query
    :param tol:
    :param method:
    :return: score, spectrum_array:n*2*2 n is same spectrum * spectrum * intent library first
    """

    return spectral_similarity.similarity(spectrum_library=ms2Library[np.argsort(ms2Library[:, 0])],
                                          spectrum_query=ms2Query[np.argsort(ms2Query[:, 0])], ms2_ppm=ppm * 1e-6,
                                          method=method, need_clean_spectra=False)