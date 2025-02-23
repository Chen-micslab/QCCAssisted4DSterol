import pandas as pd
import regex as re
import numpy as np
from emass.emass import EMass
import json
import os
import math
from Search.msdialTxtReader import msdialReader
from utils.utils import getfilepath
from tqdm import tqdm
from utils.utils import *
from typing import Union, Tuple
import re
import copy
global ppm
ppm = 1e-6


class searchMSDIALAlignExportWithMGF:
    def __init__(self, mapFilePath: str, sterolDatabase: str, alignmentPath: str,
                 mgfExportPath: str = None, msTolerance: float = 25 * ppm,
                 msmsTolerance: float = 25 * ppm, piecewiseCCSAscendingRange: list = None,
                 piecewiseRTAscendingRange: list = None,
                 finalScoreWeight: list = [0.4, 0.2, 0.4], finalScoreThreshold: float = 0.6,
                 ):

        self.txtPath = mapFilePath
        self.stDBPath = sterolDatabase
        self.alignmentPath = alignmentPath
        self.mgfPath = mgfExportPath

        self.databasePd = pd.DataFrame()
        self.featurePd = pd.DataFrame()
        self.mgfPd = pd.DataFrame()
        self.missionName: str = None
        self.missionNameDict: dict = {}
        self.missionNum: int = 0

        self.databaseMS: set = None

        self.predSTDatabasePd = pd.DataFrame()
        self.standardSTDatabasePd = pd.DataFrame()
        self.searchDatabasePd = pd.DataFrame()
        self.matchResultPd = pd.DataFrame()
        self.ms1MatchResult = pd.DataFrame()

        self.msTolerance = msTolerance
        self.msmsTolerance = msmsTolerance
        self.piecewiseCCSAscendingRange = piecewiseCCSAscendingRange
        self.piecewiseRTAscendingRange = piecewiseRTAscendingRange
        self.finalScoreWeight = finalScoreWeight
        self.finalScoreThreshold = finalScoreThreshold

        self.__init_func()

    def __init_func(self):
        self.alignmentFeature = pd.read_csv(self.alignmentPath, sep='\t')
        self.alignmentFeature.index.name = 'PeakID'
        # self.alignmentFeature = self.alignmentFeature[
        #     ['Average Rt(min)', 'Average Mz', 'Average CCS', 'Fill %']]
        df = pd.read_csv(self.stDBPath)
        self.standardSTDatabasePd = df.loc[df['Standard'] == 1]
        self.predSTDatabasePd = df.loc[df['Standard'] != 1]
        # self.alignmentFeature = self.alignmentFeature.loc[self.alignmentFeature['Average CCS'] != -1]
        # self.alignmentFeature.loc[:,'Average Rt(min)'] = self.alignmentFeature.loc[:,'Average Rt(min)']*60
        # self.alignmentFeature.rename(columns={'Average Rt(min)':'Average Rt'},inplace=True)

    def findFeatureInDB(self):
        msList = list(set(list(self.searchDatabasePd['MZ'].values)))
        msList.sort()
        self.databaseMS = set(msList)
        bar = tqdm(total=len(self.databaseMS), leave=False, dynamic_ncols=True)
        # dropNoCCSPd = self.feature.loc[self.feature['CCS'] != -1]
        matchSuccessList = []
        mzSuccessList = []
        for ms in self.databaseMS:
            # print(f'Now {ms}')
            afterMatchPd = self.alignmentFeature.loc[
                (self.alignmentFeature['Average Mz'] <= (ms * (1 + self.msTolerance))) &
                (self.alignmentFeature['Average Mz'] >= (ms * (1 - self.msTolerance)))]
            matchSuccessList = matchSuccessList + (list(afterMatchPd.index.values))
            if len(afterMatchPd) > 0:
                mzSuccessList.append(ms)
            bar.update(1)
        self.ms1MatchResult = self.searchDatabasePd.loc[self.searchDatabasePd['MZ'].isin(mzSuccessList)]
        self.matchResultPd = self.alignmentFeature.loc[self.alignmentFeature.index.isin(matchSuccessList)]
        self.databaseMS = mzSuccessList
        self.databaseMS.sort()

    def matchRTAndCCS(self):
        newIndex = 0
        bar = tqdm(total=len(self.databaseMS), leave=False, dynamic_ncols=True)
        newResultPd = pd.DataFrame(columns=list(self.matchResultPd.columns.values) + ['Theroy MZ', 'LMID'])
        for ms in self.databaseMS:
            for i, dataBaseRow in self.searchDatabasePd.loc[self.searchDatabasePd['MZ'] == ms].iterrows():
                afterMatchPd = self.matchResultPd.loc[
                    (self.alignmentFeature['Average Mz'] <= (ms * (1 + self.msTolerance))) &
                    (self.alignmentFeature['Average Mz'] >= (ms * (1 - self.msTolerance)))]
                if len(afterMatchPd) > 0:
                    for alighId, featureRow in afterMatchPd.iterrows():
                        obrt = featureRow['Average Rt(min)']*60
                        obccs = featureRow['Average CCS']
                        prert = float(dataBaseRow['RT'])
                        preccs = float(dataBaseRow['CCS'])
                        rtScore = self.piecewiseScore(observed=obrt, dataBase=prert,
                                                      ascendingRangePercent=self.piecewiseRTAscendingRange[0],
                                                      ascendingRangeAbsolute=self.piecewiseRTAscendingRange[1])
                        ccsScore = self.piecewiseScore(observed=obccs, dataBase=preccs,
                                                       ascendingRangePercent=self.piecewiseCCSAscendingRange[0],
                                                       ascendingRangeAbsolute=self.piecewiseCCSAscendingRange[1])
                        if rtScore > 0 and ccsScore > 0:
                            featureRow['LMID'] = dataBaseRow['LMID']
                            newResultPd.loc[newIndex] = featureRow
                            info = [rtScore, round(prert / 60, 2), round(ccsScore,3), preccs, ms, dataBaseRow['LMID']]
                            newResultPd.loc[
                                newIndex, ['RT Score', 'Pre RT', 'CCS Score', 'Pre CCS', 'Theroy MZ', 'LMID']] = info
                            newIndex += 1
            bar.update(1)
        self.matchResultPd = newResultPd

    def matchMSMS(self):
        for i, row in self.matchResultPd.iterrows():
            databaseRow = self.searchDatabasePd.loc[self.searchDatabasePd['LMID'] == row['LMID']]
            if int(row[self.missionName]) >= 0:
                peakIDInSingleIdx = int(row[self.missionName])
                if peakIDInSingleIdx not in list(self.featurePd['PeakID'].values):
                    # self.matchResultPd.loc[
                    #     i, f'{self.missionName}_Match MSMS'] = 'Not Find In this sample:MSdial match error?'
                    self.matchResultPd.loc[
                        i, f'{self.missionName}_Match MSMS'] = -4
                    self.matchResultPd.loc[i, f'{self.missionName}_MSMS Score'] = 0
                elif peakIDInSingleIdx in list(self.featurePd['PeakID'].values):
                    count = 0
                    msmsList = \
                        self.featurePd.loc[self.featurePd['PeakID'] == peakIDInSingleIdx, 'MSMS spectrum'].values[0]
                    # print(msmsList)
                    if msmsList != -1:
                        msmsList = msmsList.split('_')
                        msmsNumber = len(msmsList)
                        matchMSMSList = []
                        for msmsIndex in msmsList:
                            msmsIndex = int(msmsIndex)
                            msmsResult = self.msmsScore(msmsDataBaseList=databaseRow['MS2'].values[0],
                                                        msmsList=self.mgfPd.loc[msmsIndex, 'MS/MS'],
                                                        tolerance=self.msmsTolerance)
                            if msmsResult:
                                matchMSMSList += msmsResult
                                count += 1
                        msmsScore = round(count / msmsNumber, 3)
                        if msmsScore > 0:
                            self.matchResultPd.loc[i, f'{self.missionName}_Match MSMS'] = ','.join(
                                [str(x) for x in set(matchMSMSList)])
                            self.matchResultPd.loc[i, f'{self.missionName}_MSMS Score'] = msmsScore
                            self.matchResultPd.loc[i, f'{self.missionName}_Area'] = \
                                self.featurePd.loc[self.featurePd['PeakID'] == peakIDInSingleIdx, 'Area'].values[0]
                            # self.matchResultPd.loc[i, f'{self.missionName}_Area'] = \
                            #     self.featurePd.loc[self.featurePd['PeakID'] == row['Area Alignment ID'], 'Area'].values[0]
                        else:
                            # self.matchResultPd.loc[i, f'{self.missionName}_Match MSMS'] = 'Not match any MSMS'
                            self.matchResultPd.loc[i, f'{self.missionName}_Match MSMS'] = -1
                            self.matchResultPd.loc[i, f'{self.missionName}_MSMS Score'] = 0
                    else:
                        # self.matchResultPd.loc[i, f'{self.missionName}_Match MSMS'] = 'No MSMS In mgf'
                        self.matchResultPd.loc[i, f'{self.missionName}_Match MSMS'] = -2
                        self.matchResultPd.loc[i, f'{self.missionName}_MSMS Score'] = 0
            else:
                # self.matchResultPd.loc[
                #     i, f'{self.missionName}_Match MSMS'] = 'Not Find In this sample'
                self.matchResultPd.loc[
                    i, f'{self.missionName}_Match MSMS'] = -1
                self.matchResultPd.loc[i, f'{self.missionName}_MSMS Score'] = 0

    def calculateFinalScore(self):
        for i, row in self.matchResultPd.iterrows():
            finalScore = self.linearFinalScore(MSMSScore=row['MSMS Score'], RTScore=row['RT Score'],
                                               CCSScore=row['CCS Score'], weight=[0, 0.4, 0.6])
            self.matchResultPd.loc[i, 'Final Score'] = finalScore

    def selectFearture(self, threshold):
        self.matchResultPd = self.matchResultPd.loc[self.matchResultPd['MSMS Score'] != 0]
        self.matchResultPd = self.matchResultPd.loc[self.matchResultPd['RT Score'] != 0]
        self.matchResultPd = self.matchResultPd.loc[self.matchResultPd['CCS Score'] != 0]
        self.matchResultPd = self.matchResultPd.loc[self.matchResultPd['Final Score'] > threshold]
        self.matchResultPd['IsStandard'] = True
        self.matchResultPd = self.matchResultPd[
            ['LMID', 'RT (min)', 'Predict RT', 'RT Score', 'CCS', 'Predict CCS', 'CCS Score',
             'MSMS spectrum', 'MSMS Score', 'Match MSMS', 'Final Score', 'Precursor m/z', 'Theroy MZ', 'Height', 'Area',
             'S/N', 'peakId']]

    def saveResult(self, saveFolder='',appendStr: str = None):
        if saveFolder:
            savePath = os.path.join(saveFolder,f'{os.path.basename(self.alignmentPath)}_{appendStr}.csv')
        else:
            savePath = os.path.join(f'{os.path.basename(self.alignmentPath)}_{appendStr}.csv')
        self.matchResultPd.to_csv(savePath)
        print(savePath)

    def match_sterol(self, searchWhich, weight, tolerance=0.6, QC=2):
        self.searchDatabasePd = searchWhich
        self.findFeatureInDB()
        self.matchRTAndCCS()
        (txtfilepathList, txtfilenameList) = getfilepath(self.txtPath, ".csv")
        (mgfExportFilePathList, mgfExportFileNameList) = getfilepath(self.mgfPath, '.csv')
        for i, (mgf, mgfname, txt, txtname) in enumerate(
                zip(mgfExportFilePathList, mgfExportFileNameList, txtfilepathList,
                    txtfilenameList)):
            self.featurePd = pd.read_csv(txt, sep=',')
            self.featurePd['MSMS spectrum'].fillna(-1, inplace=True)
            self.mgfPd = pd.read_csv(mgf)
            baseName = os.path.basename(txtname)
            self.missionName = os.path.splitext(baseName)[0].split('_MAP')[0].rsplit('_', 1)[0]
            self.missionNameDict[self.missionName] = i
            self.missionNum = len(self.missionNameDict.keys())
            self.matchMSMS()
        for missionName in self.missionNameDict.keys():
            # print(self.matchResultPd[f"{missionName}_MSMS Score"])
            self.matchResultPd.rename(
                columns={f"{missionName}_MSMS Score": f"Sample{self.missionNameDict[missionName]}_MSMS Score"},
                inplace=True)
            self.matchResultPd.rename(
                columns={f"{missionName}_Match MSMS": f"Sample{self.missionNameDict[missionName]}_Match MSMS"},
                inplace=True)
            self.matchResultPd.rename(
                columns={f"{missionName}_Area": f"Sample{self.missionNameDict[missionName]}_MS Area"}, inplace=True)
        needColunms = ['Alignment ID', 'Fill %', 'LMID', 'Average Mz', 'Theroy MZ', 'Average Rt(min)', 'Pre RT',
                       'RT Score',
                       'Average CCS', 'Pre CCS', 'CCS Score', 'Area_mean', 'Area_std']
        MatchMSMSNameList = [f"Sample{x}_Match MSMS" for x in self.missionNameDict.values()]
        MSMSScoreNameList = [f"Sample{x}_MSMS Score" for x in self.missionNameDict.values()]
        areaNameList = [f"Sample{x}_MS Area" for x in self.missionNameDict.values()]
        for x in [MatchMSMSNameList, MSMSScoreNameList, areaNameList]:
            needColunms += x
        self.matchResultPd = self.matchResultPd[needColunms]
        for i, row in self.matchResultPd.iterrows():
            for x, y in zip(MSMSScoreNameList, self.missionNameDict.keys()):
                if row[x] > 0 and row['RT Score'] > 0 and row['CCS Score'] > 0:
                    finalScore = self.linearFinalScore(MSMSScore=row[x], RTScore=row['RT Score'],
                                                       CCSScore=row['CCS Score'], weight=weight)
                else:
                    finalScore = 0
                if finalScore >= tolerance:  # TODO:tolerance
                    self.matchResultPd.loc[i, f'Sample{self.missionNameDict[y]}_Final Score'] = finalScore
                else:
                    self.matchResultPd.loc[i, f'Sample{self.missionNameDict[y]}_Final Score'] = -1
        for i, row in self.matchResultPd.iterrows():
            totalScore = 0
            successNum = 0
            for x in range(0, self.missionNum):
                finalScore = row[f'Sample{x}_Final Score']
                if finalScore > 0:
                    totalScore += row[f'Sample{x}_Final Score']
                    successNum += 1
                else:
                    totalScore += 0
            if successNum >= QC:  # TODO:QC
                self.matchResultPd.loc[i, f'Average_Final Score'] = round(totalScore / successNum, 3)
                # self.matchResultPd.loc[i, f'Average_Final Score'] = successNum
            else:
                self.matchResultPd.loc[i, f'Average_Final Score'] = -1
        self.matchResultPd = self.matchResultPd[self.matchResultPd['Average_Final Score'] > 0]
        needAlignId = []
        for LMID, groupPd in self.matchResultPd.groupby('LMID'):
            maxScore = groupPd['Average_Final Score'].max()
            needAlignId.append(groupPd[groupPd['Average_Final Score'] == maxScore].index.values[0])
        self.matchResultPd = self.matchResultPd.loc[needAlignId, :]
        for i, row in self.matchResultPd.loc[:, areaNameList].iterrows():
            areaList = list(row.values)
            findAreaList = [x for x in areaList if x > 0]
            fillArea = min(findAreaList) / 10
            temp = []
            for x in areaList:
                if not np.isnan(x):
                    temp.append(x)
                else:
                    temp.append(fillArea)
            # self.matchResultPd.loc[i, 'Average Area'] = np.mean(temp)
            # self.matchResultPd.loc[i, 'Std Area'] = np.std(temp)

    @staticmethod
    def msmsScore(msmsDataBaseList: list, msmsList: list, tolerance: float):
        patern = r'(\d+\.\d+:\d+)'
        match = re.findall(patern, msmsList)
        matchDiaIonList = []
        if type(msmsDataBaseList) is str:
            msmsDataBaseList = msmsDataBaseList.split(',')
        elif type(msmsDataBaseList) is float:
            msmsDataBaseList = [msmsDataBaseList]
        for msms in match:
            for diaIon in msmsDataBaseList:
                msmsValue = float(msms.split(':')[0])
                diaIon = float(diaIon)
                if diaIon * (1 - tolerance) <= msmsValue <= diaIon * (1 + tolerance):
                    matchDiaIonList.append(diaIon)
        if matchDiaIonList:
            return matchDiaIonList
        else:
            return 0

    @staticmethod
    def msmsStrToDict(msmsStr: list):
        msmsDict = {}
        for msms in msmsStr:
            msmsValue = float(msms.split(':')[0])
            msmsIntent = float(msms.split(':')[-1])
            msmsDict[msmsValue] = msmsIntent
        intentSum = sum(msmsDict.values())
        for msmsValue in msmsDict.keys():
            percentIntent = msmsDict[msmsValue] / intentSum
            msmsDict[msmsValue] = percentIntent
        return msmsDict

    @staticmethod
    def piecewiseScore(observed: float, dataBase: float, ascendingRangePercent: Tuple[float, float],
                       ascendingRangeAbsolute: Tuple[float, float]):
        absError = np.abs(dataBase - observed)
        toleranceMin = dataBase * ascendingRangePercent[0] / 100 + ascendingRangeAbsolute[0]
        toleranceMax = dataBase * ascendingRangePercent[1] / 100 + ascendingRangeAbsolute[1]
        if toleranceMax <= toleranceMin:
            raise Exception('piecewiseError')
        if absError <= toleranceMin:
            return 1
        elif toleranceMin < absError <= toleranceMax:
            # print(1,(toleranceMax - absError)/(toleranceMax - toleranceMin))
            # print(2,1 - (absError - toleranceMin) / (toleranceMax - toleranceMin))
            # print(3,1 - 1 / (toleranceMax - toleranceMin) * (absError - toleranceMin))
            return 1 - 1 / (toleranceMax - toleranceMin) * (absError - toleranceMin)
        else:
            return 0

    @staticmethod
    def linearFinalScore(MSMSScore: float, RTScore: float, CCSScore: float, weight: list = [0.4, 0.2, 0.4]):
        finalScore = MSMSScore * weight[0] + RTScore * weight[1] + CCSScore * weight[2]
        return finalScore

    def merge(self, standardResult, predictResult, saveReuslt):
        standardPd = pd.read_csv(standardResult, index_col=0)
        standardPd.loc[:, 'IsStandard'] = 1
        predictPd = pd.read_csv(predictResult, index_col=0)
        alighIdInStandard = []
        LMIDInStandard = []
        for i, row in standardPd.iterrows():
            if int(row["Alignment ID"]) in list(predictPd["Alignment ID"].values):
                alighIdInStandard.append(int(row["Alignment ID"]))
            LMIDInStandard.append(row['LMID'])
        predictPd = predictPd[~(predictPd["Alignment ID"].isin(alighIdInStandard))]
        predictPd = predictPd[~(predictPd["LMID"].isin(alighIdInStandard))]
        needIdList = []
        for featureId in predictPd.groupby('Alignment ID').groups.keys():
            temp_ = predictPd[predictPd['Alignment ID'] == featureId]
            needTop = max(math.ceil(len(temp_) * 0.5), 3)
            if needTop > 3:
                needTop = 3
            id = list(temp_.nlargest(needTop, 'Average_Final Score').index)
            needIdList += id
            # print(featureId)
        resultPd = predictPd.loc[predictPd.index.isin(needIdList)]
        resultPd.loc[:, ['IsStandard']] = -1
        resultPd = pd.concat([standardPd, resultPd])
        resultPd.to_csv(saveReuslt, index=False)



if __name__ == '__main__':
    nameDict = {'B': ['brain', 1], 'F': ['feces', 1], 'C': ['cardiac', 2], 'K': ['kidney', 3], 'G': ['gut', 2],
                'H': ['hepar', 3], 'L': ['lungs', 2]}
    nameDict = {'C': ['cardiac', 2]}
    # for key in nameDict.keys():
    #     print(key)
    #     txtMapFolder = f"./tissue/{nameDict[key][0]}/MAP"
    #     mgfExportFolder = f"./tissue/{nameDict[key][0]}/DAExport"
    #     alignmentFolder = f'./tissue/{nameDict[key][0]}/alignment'
    #     filePathList, fileNameList = getfilepath(alignmentFolder,'.txt')
    #     areaID = None
    #     peakID = None
    #     for filePath in filePathList:
    #         if 'Area' in filePath:
    #             areaID = pd.read_csv(filePath,sep='\t')
    #         if 'PeakId' in filePath:
    #             peakID = pd.read_csv(filePath,sep='\t')
    #     peakID = peakID.iloc[:,[0,1,2,3,4,8,37,38,39,40]]
    #     areaID = areaID.iloc[:,[41,42]]
    #     areaID.rename(columns={'1':'Area_mean_','1.1':'Area_std_'},inplace=True)
    #     temp = pd.concat([peakID,areaID],axis=1)
    #     temp = temp.loc[temp.loc[:,'Average Mz'] <= 900]
    #     rtIndexId = list(temp.loc[temp.loc[:,'Average CCS'] == -1].index)
    #     i = 0
    #     temp[['Area Alignment ID','Area_mean','Area_std']] = None
    #     while i < len(rtIndexId)-1:
    #         temp.loc[rtIndexId[i]:(rtIndexId[i+1]-1)].loc[:, ['Area Alignment ID']] = temp.loc[rtIndexId[i],'Alignment ID']
    #         temp.loc[rtIndexId[i]:(rtIndexId[i+1]-1)].loc[:, ['Area_mean']] = round(temp.loc[rtIndexId[i], 'Area_mean_'],4)
    #         temp.loc[rtIndexId[i]:(rtIndexId[i+1]-1)].loc[:, ['Area_std']] = round(temp.loc[rtIndexId[i], 'Area_std_'],4)
    #         i += 1
    #     temp.loc[rtIndexId[i]:,:].loc[:, ['Area Alignment ID']] = temp.loc[rtIndexId[i], 'Alignment ID']
    #     temp.loc[rtIndexId[i]:,:].loc[:, ['Area_mean']] = temp.loc[rtIndexId[i], 'Area_mean_']
    #     temp.loc[rtIndexId[i]:,:].loc[:, ['Area_std']] = temp.loc[rtIndexId[i], 'Area_std_']
    #     print()
    #     temp.to_csv(f'./tissue/{nameDict[key][0]}/alignment/{key}.txt',index=0,sep='\t')
    for key in nameDict.keys():
        txtMapFolder = f"./tissue/{nameDict[key][0]}/MAP"
        mgfExportFolder = f"./tissue/{nameDict[key][0]}/DAExport"
        alignmentFolder = f'./tissue/{nameDict[key][0]}/alignment'
        searchData = searchMSDIALAlignExportWithMGF(mapFilePath=txtMapFolder,
                                                    alignmentPath=f'./tissue/{nameDict[key][0]}\\alignment\\{key}.txt',
                                                    sterolDatabase=f'../Search\\database\\STDatabaseQC{nameDict[key][1]}.csv',
                                                    mgfExportPath=mgfExportFolder,
                                                    msTolerance=25e-6, msmsTolerance=25e-6,
                                                    piecewiseRTAscendingRange=[(0, 4), (0, 0)],
                                                    piecewiseCCSAscendingRange=[(0, 1.5), (0, 0)],
                                                    )
        searchData.match_sterol(searchWhich=searchData.standardSTDatabasePd, weight=[0, 0.5, 0.5], tolerance=0)
        searchData.saveResult(appendStr='standard')
        del searchData
    #
        searchData = searchMSDIALAlignExportWithMGF(mapFilePath=txtMapFolder,
                                                    alignmentPath=f'./tissue/{nameDict[key][0]}\\alignment\\{key}.txt',
                                                    sterolDatabase=f'../Search\\database\\STDatabaseQC{nameDict[key][1]}.csv',
                                                    mgfExportPath=mgfExportFolder,
                                                    msTolerance=25e-6, msmsTolerance=25e-6,
                                                    piecewiseRTAscendingRange=[(0, 0), (84*0.5,84*2)],
                                                    piecewiseCCSAscendingRange=[(1.44*0.5, 1.44*2), (0, 0)],
                                                    )
    #
        searchData.match_sterol(searchWhich=searchData.predSTDatabasePd, weight=[0.4, 0.2, 0.4])
        searchData.saveResult(appendStr='predict')
        searchData.merge(standardResult=f"../Search/{key}.txt_standard.csv"
                         , predictResult=f"../Search/{key}.txt_predict.csv",
                         saveReuslt=f'./result/{key}-result.csv')


