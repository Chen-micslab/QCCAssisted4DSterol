import pandas as pd
import random
from tqdm import tqdm
# from tqdm.notebook import tqdm
import time
import os
from multiprocessing import Pool, Lock
import copy
import regex
from utils.utils import calculate_CCS, getfilepath


def getMsMsStr(msmsContent: list):
    if len(msmsContent) == 0:
        return 'None'
    else:
        msmsStr = ''.join(msmsContent)
        msmsStr = msmsStr.replace('\t\n', '\n')
        msmsStr = msmsStr.replace('\t', ':')
    return msmsStr


def readBrukerDAMgf(i, mgfFile, attribute):
    attributeList = attribute
    attributeWithMSMS = copy.deepcopy(attribute)
    attributeWithMSMS.append('MS/MS')
    attributeWithMSMS.append('MOBILITY')
    attributeWithMSMS.append('CCS')
    attributeWithMSMS.append('INTENSITY')
    df = pd.DataFrame(columns=attributeWithMSMS)
    attributeDict = {x: None for x in attributeWithMSMS}
    endChar = "END IONS\n"
    count = 0
    attributeLen = len(attribute)
    tempMCSVName = str(random.randint(100000, 999999))
    bar = tqdm(total=len(mgfFile), leave=False,
               desc='Mission:' + str(i) + ' pid:' + str(os.getpid()), position=i,
               dynamic_ncols=True)
    while 1:
        try:
            mgfFile.index(endChar)
        except:
            break
        if '#charge undefined' not in mgfFile[0]:
            for line, attr in zip(mgfFile[5:attributeLen + 5], attributeList):
                splitResult = line.split(attr + '=')
                if attr == 'TITLE':
                    pattern = r"1/K0=\d.\d\d\d"
                    k0 = regex.findall(pattern, splitResult[-1])[0].split('=')[-1]
                    attributeDict['MOBILITY'] = float(k0)
                if attr == 'PEPMASS':
                    ms = splitResult[-1].split('\t')[0]
                    inten = splitResult[-1].split('\t')[-1]
                    attributeDict['PEPMASS'] = float(ms)
                    attributeDict['INTENSITY'] = float(inten)
                    continue
                attributeDict[attr] = splitResult[-1].rstrip('\n')
            attributeDict['CCS'] = round(calculate_CCS(attributeDict['PEPMASS'], attributeDict['MOBILITY'], 1), 2)
            endIndex = mgfFile.index(endChar)
            msmsContent = mgfFile[attributeLen + 5:endIndex]
            attributeDict['MS/MS'] = getMsMsStr(msmsContent)
            del mgfFile[0:endIndex + 2]
            df.loc[0] = attributeDict
            bar.update(endIndex + 2)
            count += 1
            if os.path.exists(tempMCSVName + '.mcsv'):
                df.to_csv(tempMCSVName + '.mcsv', mode='a', header=False)
            else:
                df.to_csv(tempMCSVName + '.mcsv', mode='a')
        else:
            endIndex = mgfFile.index(endChar)
            del mgfFile[0:endIndex + 2]
            df.loc[0] = attributeDict
            bar.update(endIndex + 2)
    return tempMCSVName


def readMgf(i, mgfFile, attribute):
    attributeList = attribute
    attributeWithMSMS = copy.deepcopy(attribute)
    attributeWithMSMS.append('MS/MS')
    df = pd.DataFrame(columns=attributeWithMSMS)
    attributeDict = {x: None for x in attributeWithMSMS}
    endChar = "END IONS\n"
    count = 0
    attributeLen = len(attribute)
    tempMCSVName = str(random.randint(100000, 999999))
    bar = tqdm(total=len(mgfFile), leave=False,
               desc='Mission:' + str(i) + ' pid:' + str(os.getpid()), position=i,
               dynamic_ncols=True)
    while 1:
        try:
            mgfFile.index(endChar)
        except:
            break
        for line, attr in zip(mgfFile[1:attributeLen + 1], attributeList):
            splitResult = line.split(attr + '=')
            attributeDict[attr] = splitResult[-1].rstrip('\n')
        endIndex = mgfFile.index(endChar)
        msmsContent = mgfFile[attributeLen + 1:endIndex]
        attributeDict['MS/MS'] = getMsMsStr(msmsContent)
        del mgfFile[0:endIndex + 2]
        df.loc[0] = attributeDict
        bar.update(endIndex + 2)
        count += 1
        if os.path.exists(tempMCSVName + '.mcsv'):
            df.to_csv(tempMCSVName + '.mcsv', mode='a', header=False)
        else:
            df.to_csv(tempMCSVName + '.mcsv', mode='a')
    return tempMCSVName


class mgfReader:
    def __init__(self, mgfFilePath: str, cpuNum: int = 7, isBruker: bool = False,
                 saveFileName: str = None):

        self.mgf: list = None
        self.splitMgfList: list = []
        self.mgfFileName = os.path.basename(mgfFilePath)
        if not saveFileName:
            self.exportCSVName = os.getcwd() + self.mgfFileName.rsplit('.', 1)[0] + '_DAExport.csv'
        else:
            self.exportCSVName = saveFileName + '/' + self.mgfFileName.rsplit('.', 1)[0] + '_DAExport.csv'
        self.mgfFilePath: str = mgfFilePath
        self.isBrukerDaMgfFile = isBruker

        self.attribute: list = []
        self.attributeLen: int = 0

        self.tempMCSVFileNameList: list = []
        self.indexList: list = None
        self.mgfPd: pd.core.DataFrame = None
        self.osCpuNum: int = os.cpu_count()
        self.useProcessNum: int = cpuNum
        self._init_func()
        self.final_csvPd = pd.DataFrame()

    def _init_func(self):
        self._deleteAllMCSV()
        self._readMgfFile(self.mgfFilePath)
        self._splitMgf()
        self._findAttribute()
        if self.useProcessNum >= self.osCpuNum:
            self.useProcessNum = self.osCpuNum - 1

    def start(self):
        self.multiProcessRead()
        self.to_sumCSV(self.exportCSVName)

    def _readMgfFile(self, mgfFilePath):
        with open(mgfFilePath, mode='r') as mgfFile:
            self.mgf = mgfFile.readlines()

    def _findAttribute(self):
        startChar = "BEGIN IONS\n"
        endChar = "END IONS\n"
        startIndex = self.mgf.index(startChar)
        endIdx = self.mgf.index(endChar)
        for line in self.mgf[startIndex:endIdx]:
            splitResult = line.split('=')
            if len(splitResult) > 1:
                self.attribute.append(splitResult[0])
        self.attributeLen = len(self.attribute)  # without ms/ms
        try:
            self.attribute.index('CHARGE')
        except:
            self.attribute.append('CHARGE')
        self.mgfPd = pd.DataFrame(columns=self.attribute)

    def _splitMgf(self):
        if self.useProcessNum == 1:
            self.splitMgfList = [self.mgf]
        elif len(self.mgf) < 5000:
            self.splitMgfList = [self.mgf]
            self.useProcessNum = 1
        else:
            endChar = "END IONS\n"
            mgfContent = self.mgf
            mgfContentLength = len(self.mgf)
            splitMgfContentLength = mgfContentLength // self.useProcessNum
            for i in range(0, self.useProcessNum):
                splitTemp = mgfContent[0:splitMgfContentLength]
                splitTemp.reverse()
                # print(splitTemp[0])
                idx = splitTemp.index(endChar)
                self.splitMgfList.append(mgfContent[0:splitMgfContentLength - idx + 1])
                del mgfContent[0:splitMgfContentLength - idx + 1]
        if self.isBrukerDaMgfFile:
            startIdx = self.splitMgfList[0].index('BEGIN IONS\n')
            self.splitMgfList[0] = self.splitMgfList[0][startIdx - 4:]

    def multiProcessRead(self):
        print(f'Use {self.useProcessNum} Process ')
        time.sleep(2)
        startTime = time.time()
        processResultList = []
        with Pool(processes=self.useProcessNum, initializer=tqdm.set_lock, initargs=(Lock(),), ) as p:
            for i in range(0, self.useProcessNum):
                if self.isBrukerDaMgfFile:
                    result = p.apply_async(readBrukerDAMgf, (i, self.splitMgfList[i], self.attribute))
                else:
                    result = p.apply_async(readMgf, (i, self.splitMgfList[i], self.attribute))
                processResultList.append(result)
            p.close()
            p.join()
            for result in processResultList:
                self.tempMCSVFileNameList.append(result.get())
        endTime = time.time()
        print(f'Total time {endTime - startTime}')
        # processes = []
        # for i in range(0, self.useProcessNum):
        #     readProcess = Process(target=readMgf, args=(i, self.splitMgfList[i], self.attribute,))
        #     readProcess.start()
        #     processes.append(readProcess)
        # for _ in processes:
        #     readProcess.join()

    def to_sumCSV(self, savepath):
        self.final_csvPd = pd.DataFrame()
        for tempMCSVFileName in self.tempMCSVFileNameList:
            if len(self.final_csvPd) == 0:
                self.final_csvPd = pd.read_csv(tempMCSVFileName + '.mcsv')
            else:
                tempPd = pd.read_csv(tempMCSVFileName + '.mcsv')
                self.final_csvPd = pd.concat([self.final_csvPd, tempPd])
            os.remove(tempMCSVFileName + '.mcsv')
        self.final_csvPd = self.final_csvPd.drop(axis=1, columns=['Unnamed: 0'])
        self.standardCSV()
        self.final_csvPd['Index'] = range(0, len(self.final_csvPd))
        self.final_csvPd.to_csv(savepath, index=False)
        print(f'Save output in {savepath}')

    def standardCSV(self):
        if self.isBrukerDaMgfFile:
            self.final_csvPd = self.final_csvPd.drop(axis=1, columns=['TITLE', 'RAWSCANS', 'INTENSITY'])
            self.final_csvPd['RTINSECONDS'] = self.final_csvPd['RTINSECONDS']
        else:
            self.final_csvPd = self.final_csvPd.drop(axis=1, columns=['TITLE', 'SCANS', 'ION'])
            self.final_csvPd['RTINMINUTES'] = self.final_csvPd['RTINMINUTES'] * 60
            self.final_csvPd = self.final_csvPd.rename(columns={'RTINMINUTES': 'RTINSECONDS'})

    @staticmethod
    def _deleteAllMCSV():
        curPath = os.path.abspath('.')
        MCSVPath, MCSVName = getfilepath(curPath, '.mcsv')
        for path in MCSVPath:
            os.remove(path)
            print(f'Remove {path}')
        # print(curPath)


if __name__ == '__main__':
    # --------------------read mgfReader ----------------------------------
    mgfFolder = "./test"  # input a folder path contain mgf file from DA(Bruker DataAnalysis) convert results
    (filepathList, filenameList) = getfilepath(mgfFolder, ".mgf")
    for filepath, filename in zip(filepathList, filenameList):
        print(f'Now read {filename}')
        myReader = mgfReader(filepath, cpuNum=7, isBruker=True)
        myReader.start()
        del myReader
