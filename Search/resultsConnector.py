from Search.msdialTxtReader import msdialReader
from typing import Dict, List
from tqdm import tqdm
from pandas.core.frame import DataFrame
from multiprocessing import Pool, Lock, Manager
from utils.utils import getfilepath
import os
import time
import pandas as pd


def matchPeaks(mgfPd, i, peakListPd, msTolerance, mapDict):
    mgfPd = mgfPd.sort_values('RTINSECONDS')
    with tqdm(total=len(mgfPd), position=i, dynamic_ncols=True,
              leave=False, desc='Mission:' + str(i) + ' pid:' + str(os.getpid())) as bar:
        for idx, row in mgfPd.iterrows():
            rtInMin = row['RTINSECONDS'] / 60.0
            pepMass = row['PEPMASS']
            mobility = row['MOBILITY']
            index = row['Index']
            matchRTPd = peakListPd.loc[(peakListPd['RT left(min)'] <= rtInMin)
                                       & (peakListPd['RT right (min)'] >= rtInMin)]

            matchRTAndMSPd = matchRTPd.loc[(peakListPd['Precursor m/z'] * (1 - msTolerance) <= pepMass) &
                                           (peakListPd['Precursor m/z'] * (1 + msTolerance) >= pepMass)]
            matchRTAndMSAndMobilityPd = matchRTAndMSPd.loc[(peakListPd['Mobility left'] <= mobility) &
                                                           (peakListPd['Mobility right'] >= mobility)]
            b = matchRTAndMSAndMobilityPd.index.values
            if len(b) > 0:
                if b[0] not in mapDict.keys():
                    mapDict[b[0]] = str(index)
                else:
                    mapDict[b[0]] += '_{}'.format(index)
            bar.update(1)
        return mapDict


class Connector(msdialReader):
    def __init__(self, txtFilePath: str, mgfExportFilePath: str, maxChage: int = 1, addIon: str = '[M+H]+'
                 , cpuNum: int = 5, savepath: str = None):
        super(Connector, self).__init__(txtFilePath, maxChage, addIon)
        self.mgfFilePd = pd.read_csv(mgfExportFilePath)
        self.splitMgfPdList: List[DataFrame] = []
        self.msTolerance = 25e-6
        self.mapDict: Dict[int, List[int]] = {}
        self.mgffileName = os.path.basename(mgfExportFilePath)
        self.osCpuNum: int = os.cpu_count()
        self.useProcessNum: int = cpuNum
        if not savepath:
            self.exportCSVName = os.getcwd() + self.mgffileName.rsplit('_', 1)[0] + '_MAP.csv'
        else:
            self.exportCSVName = savepath + '/' + self.mgffileName.rsplit('_', 1)[0] + '_MAP.csv'
        self.__init_func()

    def __init_func(self):
        if self.useProcessNum >= self.osCpuNum:
            self.useProcessNum = self.osCpuNum - 1
        self._splitMgf()

    def _splitMgf(self):
        if self.useProcessNum == 1:
            self.splitMgfPdList = [self.mgfFilePd]
        elif len(self.mgfFilePd) < 5000:
            self.splitMgfPdList = [self.mgfFilePd]
            self.useProcessNum = 1
        else:
            mgfPd = self.mgfFilePd
            mgfPdLength = len(self.mgfFilePd)
            splitMgfPdLength = (mgfPdLength // self.useProcessNum) + 1
            for i in range(0, self.useProcessNum):
                self.splitMgfPdList.append(mgfPd.iloc[i * splitMgfPdLength:(i + 1) * splitMgfPdLength, :])

    def exportMap(self):
        mapPd = pd.DataFrame.from_dict(self.mapDict, orient='index')
        mapfileName = self.txtfileName.split('.')[0] + '_map.csv'
        mapPd.to_csv(mapfileName)

    def exportTxtWithMap(self):
        df = self.peakListPd.drop('MSMS spectrum', axis=1)
        df['MSMS spectrum'] = ''
        for key in self.mapDict.keys():
            df.loc[key, 'MSMS spectrum'] = self.mapDict[key]
        df.to_csv(self.exportCSVName)
        print(f'Save Output in {self.exportCSVName}')

    def multiProcessConvert(self):
        print(f'Use {self.useProcessNum} Process ')
        time.sleep(0.5)
        startTime = time.time()
        processResultList = []
        mapDict = Manager().dict()
        with Pool(processes=self.useProcessNum, initializer=tqdm.set_lock, initargs=(Lock(),), ) as p:
            for i in range(0, self.useProcessNum):
                result = p.apply_async(matchPeaks,
                                       (self.splitMgfPdList[i], i, self.peakListPd, self.msTolerance, mapDict))
                processResultList.append(result)
            p.close()
            p.join()
            # for result in processResultList:
            #     self.tempMCSVFileNameList.append(result.get())
        endTime = time.time()
        print(f'Total time {endTime - startTime}')
        self.mapDict = mapDict

