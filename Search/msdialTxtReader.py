import pandas as pd
from pandas.core.frame import DataFrame
from typing import List
import os


class msdialReader:
    def __init__(self, filePath: str, maxChage: int = 1, addIon: str = '[M+H]+', sep: str = '\t'
                 , savepath: str = None):
        self.txtFilePath: str = os.path.basename(filePath)
        self.savepath: str = savepath
        if not savepath:
            self.exportCSVName = os.getcwd() + self.txtFilePath.rsplit('.', 1)[0] + '_MSDIALEXPORT.csv'
        else:
            self.exportCSVName = self.savepath + '/' + self.txtFilePath.rsplit('.', 1)[0] + '_MSDIALEXPORT.csv'
        try:
            self.peakListPd: DataFrame = pd.read_csv(filePath, sep=sep, index_col=0)
        except:
            raise Exception('Can not read txt')
        self.attributeNameList: List[str] = ['RT left(min)', 'RT (min)', 'RT right (min)',
                                             "Mobility left", 'Mobility', 'Mobility right',
                                             "CCS", 'Precursor m/z', 'Height',
                                             "Area", 'S/N', 'MSMS spectrum']
        if 'Adduct' in self.peakListPd.columns:
            self.peakListPd = self.peakListPd.loc[self.peakListPd['Adduct'] == addIon]
            self.peakListPd = self.peakListPd.loc[self.peakListPd['Isotope'] == 'M + 0']
        self.peakListPd = self.peakListPd.loc[:, self.attributeNameList]
        self.txtfileName = os.path.basename(filePath)

    def saveSimpleMsdialTxt(self):
        self.peakListPd.to_csv(self.exportCSVName)
        print(f'Save output in {self.exportCSVName}')