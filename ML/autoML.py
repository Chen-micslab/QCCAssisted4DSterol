"""
Machine learning module for regression based sklearn
==================================
author: Sun Jian
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_squared_error
from sklearn.decomposition import PCA as PCA
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import time
import pickle as pkl


class autoML:
    def __init__(self, reactionData, saveFilePath: str, seedStartRange=0, seedEndRange=5, seedStep=1,
                 test_ratio=0.2, cvNumber=10, score=[], splitRule=None):
        self.seedStart = seedStartRange
        self.seedEnd = seedEndRange
        self.seedStep = seedStep
        self.seedRange = np.arange(self.seedStart, self.seedEnd, self.seedStep)
        self.seedRangeChar = str(self.seedStart) + '_' + str(self.seedEnd) + '_' + str(self.seedStep)
        self.seed: int = None
        self.testRatio = test_ratio
        self.printDf = pd.DataFrame(index=self.seedRange)
        if score is None:
            self.needScoreList = ['RMSE', 'R2', 'MAE']
        elif isinstance(score, list):
            self.needScoreList = score
        else:
            raise Exception(f'wrong score input need list or none')
        self.saveFileFolderPath = saveFilePath
        self.cvNumber = cvNumber
        self.featureTarget = pd.DataFrame
        self.feature = pd.DataFrame
        self.target = pd.DataFrame
        self.cv = KFold(n_splits=5, shuffle=False, random_state=None)
        self.testResult = {}
        self.result = {}
        self.resultList = []
        self.gridSearch_Metric: str = 'neg_root_mean_squared_error'
        self.modelNumber: int
        self.modelName: str
        self.grid = {}
        self.predictError = []
        self.calculateFunction: int
        self.y_predict = []
        self.y_test = []
        self.X_test = []
        self.best_estimator = []
        self.baseline = float
        self.outPred = None
        self.outObserve = None
        self.splitRule = splitRule
        self.finalModel = None
        self.predictPercentError = 0
        self.verboseResult = pd.DataFrame(columns=['Trial', 'params', 'mean_score'])
        assert reactionData is not None, 'Need data'
        if isinstance(reactionData, pd.core.frame.DataFrame):
            self.reactionData = reactionData
        elif isinstance(reactionData, str):
            self.reactionData = pd.read_csv(reactionData, encoding="unicode_escape", low_memory=False)
        else:
            raise Exception(f'Check Reaction Datatype')

    def analyzeReactionData(self, featureIndexList: list, targetIndex: int):
        print('-----------------------analyzeReactionData-----------------------------\n')
        self.featureTarget = pd.DataFrame(index=self.reactionData.index)
        featureIndexList_ = []
        for x in featureIndexList:
            if isinstance(x, int):
                featureIndexList_.append(x)
                featureIndexList_.append(x + 1)
            elif isinstance(x, list):
                for a in x:
                    if not isinstance(a, int):
                        raise Exception(f'Need list.Ex:[1,5] or [1,5,[7,8]].')
                    else:
                        None
                featureIndexList_.append(x[0])
                featureIndexList_.append(x[-1])
            else:
                raise Exception(f'Need list.Ex:[1,5] or [1,5,[7,8]].')
        if sorted(featureIndexList_) != featureIndexList_:
            raise Exception(f'Check list.Need a absolute incremental input list.Ex:[1,5] or [1,5,[7,8]].'
                            f'Now is {featureIndexList}')
        for i in range(0, len(featureIndexList_), 2):
            self.feature = self.reactionData.iloc[:, featureIndexList_[i]:featureIndexList_[i + 1]]
        featureNumber = len(self.feature.columns)
        self.target = self.reactionData.iloc[:, targetIndex:targetIndex + 1]
        self.featureTarget = pd.concat([self.target, self.feature], axis=1)
        self.baseline = self.target.mean()
        print('Analyze Reaction Data Success\n')
        print(f'Feature Number is {featureNumber}\n')
        print(f'Feature is {self.feature.columns}\n')
        print('Predict Target Name: {targetName}\n'.format(targetName=self.reactionData.iloc
        [:, targetIndex:targetIndex + 1].columns.values[0]))
        print(f'Get Reaction Data. Now Please Set The CV Parameters\n')
        print(f'total reaction number is {len(self.target)}\n')
        print('-----------------------analyzeReactionDataEnd--------------------------\n')
        return self

    def modelSelectionInput(self):
        print(f'Input the number to choose the model')
        print(f'1.RF\n')
        print(f'2.XGBoost\n')
        print(f'3.Adaboost\n')
        print(f'4.GBDT\n')
        print(f'5.CatBoost\n')
        print(f'6.LightGBM\n')
        print(f'7.SVM\n')
        print(f'8.Lasso\n')
        print(f'9.Ridge\n')
        print(f'10.ElasticNet\n')
        print(f'11.KNN\n')
        print(f'0.use for debug RF\n')
        function_Number = input("Model Number:")
        assert function_Number in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], 'Number'
        self.modelNumber = function_Number

    def gridGenerator(self, gridParameter):
        if gridParameter is None:
            pass
        else:
            self.grid = gridParameter

    def seedGenerator(self, seedList):
        if seedList is None:
            pass
        else:
            self.seed = seedList

    def _chooseModel(self, seed):
        rng = np.random.RandomState(seed)
        if 1 == int(self.modelNumber):
            self.modelName = 'RandomForest'
            if not self.grid:
                self.grid = [{
                    'regressor__n_estimators': [100, 200, 500, 1000]
                }]
            model = Pipeline(  # pipeline   可以防止数据泄露
                [
                    ("regressor", RandomForestRegressor(random_state=seed))
                ]
            )
            return model
        elif 2 == int(self.modelNumber):
            self.modelName = 'XgBoost'
            if not self.grid:
                self.grid = [{
                    'regressor__booster': ['gbtree'],
                    'regressor__max_depth': [5],
                    'regressor__n_estimators': [1000],
                    'regressor__learning_rate': [0.01],
                }]
            model = Pipeline(
                [
                    ("regressor", XGBRegressor(random_state=rng))
                ]
            )
            return model
        elif 3 == int(self.modelNumber):
            self.modelName = 'AdaBoost'
            if not self.grid:
                self.grid = [{
                    'regressor__n_estimators': [1000],
                    'regressor__learning_rate': [0.01, 0.1, 0.5],
                    'regressor__loss': ['linear', 'square']
                }]
            model = Pipeline(
                [
                    ("regressor", AdaBoostRegressor(random_state=rng))
                ]
            )
            return model
        elif 4 == int(self.modelNumber):
            self.modelName = 'GBDT'
            if not self.grid:
                self.grid = [{
                    'regressor__n_estimators': range(100, 2100, 300),
                    'regressor__subsample': [round(x, 2) for x in np.arange(0.1, 1.3, 0.3)],
                    'regressor__alpha': [0.001, 0.0001]
                }]
            model = Pipeline(
                [
                    ("regressor", GradientBoostingRegressor(random_state=rng))
                ]
            )
            return model
        elif 5 == int(self.modelNumber):
            self.modelName = 'CatBoost'
            if not self.grid:
                self.grid = [{
                    'regressor__learning_rate': [0.09, 0.1, 0.01],  # onehot
                    'regressor__depth': [5, 7, 9],  # 3,5,7
                    'regressor__n_estimators': [1000],
                    # 'regressor__iterations': [1000],
                }]
            model = Pipeline(
                [
                    ("regressor", CatBoostRegressor(
                        random_seed=seed, task_type="CPU", verbose=5, thread_count=7))
                ]
            )
            return model
        elif 6 == int(self.modelNumber):
            self.modelName = 'LightGBM'
            if not self.grid:
                self.grid = [{
                    'regressor__n_estimators': [1000],
                    'regressor__learning_rate': [0.09, 0.1, 0.11],
                    'regressor__max_depth': [5, 7, 9]
                }]
            model = Pipeline(
                [
                    ("regressor", LGBMRegressor(random_seed=seed))
                    # lightGBM feature name cannot appear ',' , '[' , ']' , '{' , '}' , '"' , ':' character.
                    # Because these characters have special meaning in JSON file used by lightGBM module.
                ]
            )
            return model
        elif 7 == int(self.modelNumber):
            pca = PCA(n_components=0.85, random_state=seed)
            self.modelName = 'SVM'
            if not self.grid:
                self.grid = [{
                    # 'regressor__kernel': ['rbf'],
                    'regressor__kernel': ['rbf', 'linear'],
                    'regressor__C': [0.001, 0.01, 0.1, 1, 10],
                    'regressor__gamma': [0.001, 0.01, 0.1, 1, 10],
                }]
            # model = Pipeline(
            #     [("standard", StandardScaler()),
            #      ("PCA", pca),
            #      ("regressor", SVR())
            #      ]
            # )
            model = Pipeline(
                [("standard", StandardScaler()),
                 ("regressor", SVR())
                 ]
            )
            return model
        elif 8 == int(self.modelNumber):
            pca = PCA(n_components=0.1, random_state=seed)
            self.modelName = 'Lasso'
            if not self.grid:
                self.grid = [{
                    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
                }]
            model = Pipeline(
                [("standard", StandardScaler()),
                 ("regressor", Lasso(max_iter=10000))
                 ]
            )
            return model
        elif 9 == int(self.modelNumber):
            pca = PCA(n_components=0.99, random_state=seed)
            self.modelName = 'Ridge'
            if not self.grid:
                self.grid = [{
                    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
                }]
            model = Pipeline(
                [("standard", StandardScaler()),
                 # ("PCA", pca),
                 ("regressor", Ridge(max_iter=10000))
                 ]
            )
            return model
        elif 10 == int(self.modelNumber):
            pca = PCA(n_components=0.99, random_state=seed)
            self.modelName = 'ElasticNet'
            if not self.grid:
                self.grid = [{
                    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9],
                    'regressor__l1_ratio': [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]
                }]
            model = Pipeline(
                [("standard", StandardScaler()),
                 # ("PCA", pca),
                 ("regressor", ElasticNet(random_state=seed, max_iter=10000))
                 ]
            )
            return model
        elif 11 == int(self.modelNumber):
            self.modelName = 'KNN'
            if not self.grid:
                self.grid = [{
                    'regressor__n_neighbors': [3, 5, 7, 9]
                }]
            model = Pipeline(
                [
                    # ("PCA", pca),
                    ("regressor", KNeighborsRegressor())
                ]
            )
            return model
        elif 0 == int(self.modelNumber):
            self.modelName = 'debugRandomForest'
            self.grid = [{
                'regressor__n_estimators': [2]
            }]
            model = Pipeline(
                [
                    ("regressor", RandomForestRegressor(random_state=seed))
                ]
            )
            return model
        else:
            None

    def ML_GridSearch(self, seed, gridSearch_Metric='neg_root_mean_squared_error') -> None:
        self.seed = seed
        self.gridSearch_Metric = gridSearch_Metric
        rng = np.random.RandomState(seed)
        print(f'now seed is {self.seed}')

        X_train, X_test, y_train, y_test = train_test_split(self.feature, self.target.values.ravel(),
                                                            test_size=self.testRatio, random_state=rng)
        self.X_test = X_test
        self.y_test = y_test
        estimator = self._chooseModel(self.seed)
        gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                          scoring=gridSearch_Metric, cv=self.cv, verbose=5, n_jobs=7)
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        self.getCVVerboseResult(self.seed, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
        print()
        self.y_predict = y_pred
        self.result = self.calculateTestResultMetric(self.y_test, self.y_predict)
        self.result['Trial'] = self.seed
        self.result['best_parameter'] = gs.best_params_
        self.result['best_train_score'] = -gs.best_score_
        self.resultList.append(self.result)
        print(f'Grid_search:{gs.best_params_}')
        return None

    def ML_GridSearchLOO(self, seed, gridSearch_Metric='neg_root_mean_squared_error') -> None:
        self.seed = seed
        self.gridSearch_Metric = gridSearch_Metric
        rng = np.random.RandomState(seed)
        print(f'now seed is {self.seed}')
        feartureDrop = self.feature.drop(index=self.feature.index[seed])
        targetDrop = self.target.drop(index=self.feature.index[seed])
        X_train, X_test, y_train, y_test = train_test_split(feartureDrop, targetDrop.values.ravel(),
                                                            test_size=self.testRatio, random_state=rng)
        self.X_test = X_test
        self.y_test = y_test
        estimator = self._chooseModel(self.seed)
        gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                          scoring=gridSearch_Metric, cv=self.cv, verbose=5, n_jobs=7)
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        self.getCVVerboseResult(self.seed, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
        print()
        self.y_predict = y_pred
        self.result = self.calculateTestResultMetric(self.y_test, self.y_predict)
        self.result['Trial'] = self.seed
        self.result['best_parameter'] = gs.best_params_
        self.result['best_train_score'] = -gs.best_score_
        self.resultList.append(self.result)
        L00pred = gs.best_estimator_.predict(self.feature.iloc[[seed], :])
        self.result['XLOO'] = self.target.values[seed][0]
        self.result['YLOO'] = L00pred[0]
        self.result['LOO_Relative_Error'] = (abs(self.result['XLOO'] - self.result['YLOO']) / self.result['XLOO'])
        print(f'LOO pred {L00pred}')
        print(f'Grid_search:{gs.best_params_}')
        return None

    def getCVVerboseResult(self, trial, params, meantest):
        oneVerboseResult = pd.DataFrame()
        oneVerboseResult['Trial'] = [str(trial)] * len(params)
        oneVerboseResult['params'] = params
        oneVerboseResult['mean_score'] = meantest
        self.verboseResult = pd.concat([self.verboseResult, oneVerboseResult], axis=0)
        print()

    def vsBaseLine(self):
        print('baseline\n')
        # baseLineList = np.ones(self.y_test.shape)
        baseLineList = self.baseline.values[0] * np.ones(self.y_test.shape)
        baseLine_result = self.calculateTestResultMetric(baseLineList, self.y_test)
        # print(baseLine_result['rmse'])
        self.result['base_rmse'] = baseLine_result['rmse']
        self.result['base_r2'] = baseLine_result['r2']

    def LooScore(self,gridSearch_Metric='neg_root_mean_squared_error',verbose:bool=False,addChar:str=''):
        loo = LeaveOneOut()
        MREscore = []
        predict = []
        # Iterate over each LOO fold
        for i, (train_index, test_index) in enumerate(loo.split(self.feature, self.target.values.ravel())):
            # Get training and test sets
            X_train, X_test = self.feature.values[train_index], self.feature.values[test_index]
            y_train, y_test = self.target.values.ravel()[train_index], self.target.values.ravel()[test_index]
            estimator = self._chooseModel(self.seed)
            gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                              scoring=gridSearch_Metric, cv=5, verbose=5, n_jobs=7)
            gs.fit(X_train, y_train)
            y_pred = gs.best_estimator_.predict(X_test)
            self.getCVVerboseResult(i, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
            self.result['best_parameter'] = gs.best_params_
            self.result['best_train_score'] = -gs.best_score_
            self.resultList.append(self.result)
            a = np.abs(y_pred - y_test) / y_test
            print(f'ytest{y_test} y_pred{y_pred} relative error{a[0]:.2%}')
            MREscore.append(a[0])
            predict.append(y_pred[0])
        if verbose:
            predDetails = pd.DataFrame(
                [self.target.values.ravel(), predict, MREscore],
                index=['observed', 'predicted', 'error']).T
            predDetails.to_csv(f'{self.saveFileFolderPath}/LOO.csv')
        fileName = 'seedRange' + '_' + self.seedRangeChar + '_model_' + self.modelName + '_testRadio_' + \
                   str(self.testRatio) + '_' + str(self.gridSearch_Metric) + addChar
        self.printDf = pd.DataFrame(self.resultList)
        verboseResult = self.saveFileFolderPath + '/' + fileName + '_verbose' + '.csv'
        self.verboseResult.to_csv(verboseResult)
        df = pd.read_csv(verboseResult, index_col=0)
        df.groupby('params').mean().to_csv(verboseResult)
        print(f'{np.mean(MREscore):.2%}')
        print(f'{np.median(MREscore):.2%}')


    def LooScoreWithAssisted(self, assitedPd,gridSearch_Metric='neg_root_mean_squared_error',verbose:bool=False,
                             addChar: str=''):
        loo = LeaveOneOut()
        MREscore = []
        predict = []
        assistedFeature = assitedPd.values[:, 1:]
        assistedCCS = assitedPd.values[:, 0:1]
        # Iterate over each LOO fold
        for i, (train_index, test_index) in enumerate(loo.split(self.feature, self.target.values.ravel())):
            # Get training and test sets
            X_train, X_test = self.feature.values[train_index], self.feature.values[test_index]
            y_train, y_test = self.target.values.ravel()[train_index], self.target.values.ravel()[test_index]
            X_train = np.concatenate((X_train, assistedFeature))
            y_train = np.append(y_train, assistedCCS)
            estimator = self._chooseModel(self.seed)
            gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                              scoring=gridSearch_Metric, cv=self.cv, verbose=5, n_jobs=7)
            gs.fit(X_train, y_train)
            y_pred = gs.best_estimator_.predict(X_test)
            self.getCVVerboseResult(i, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
            self.result['best_parameter'] = gs.best_params_
            self.result['best_train_score'] = -gs.best_score_
            self.resultList.append(self.result)
            a = np.abs(y_pred - y_test) / y_test
            print(f'ytest{y_test} y_pred{y_pred} relative error{a[0]:.2%}')
            MREscore.append(a[0])
            predict.append(y_pred[0])
        if verbose:
            predDetails = pd.DataFrame(
                [self.target.values.ravel(), predict, MREscore],
                index=['observed', 'predicted', 'error']).T
            predDetails.to_csv(f'{self.saveFileFolderPath}/LOOAssist.csv')
        fileName = 'seedRange' + '_' + self.seedRangeChar + '_model_' + self.modelName + '_testRadio_' + \
                   str(self.testRatio) + '_' + str(self.gridSearch_Metric) + addChar
        self.printDf = pd.DataFrame(self.resultList)
        verboseResult = self.saveFileFolderPath + '/' + fileName + '_verbose' + '.csv'
        self.verboseResult.to_csv(verboseResult)
        df = pd.read_csv(verboseResult, index_col=0)
        df.groupby('params').mean().to_csv(verboseResult)
        print(f'{np.mean(MREscore):.2%}')
        print(f'{np.median(MREscore):.2%}')


    def _finalModel(self, gridSearch_Metric='neg_root_mean_squared_error'):
        if self.finalModel is None:
            seed = 0
            gs = GridSearchCV(estimator=self._chooseModel(seed), param_grid=self.grid,
                              scoring=gridSearch_Metric, cv=KFold(n_splits=5, shuffle=True, random_state=seed),
                              verbose=3, n_jobs=7)
            gs.fit(self.feature, self.target.values.ravel())
            print(f'final_model:{gs.best_params_}')
            self.finalModel = gs.best_estimator_.fit(self.feature, self.target.values.ravel())
        else:
            self.finalModel.fit(self.feature, self.target.values.ravel())

    def outSamplePredict(self, outData, targetIndex, featureStartIndex, featureEndIndex,
                         gridSearch_Metric='neg_root_mean_squared_error',
                         additiveNameChar=None):
        print(f'out sample')
        seed = 1
        if isinstance(outData, str):
            out = pd.read_csv(outData, encoding="unicode_escape", low_memory=False)
        elif isinstance(outData, pd.core.frame.DataFrame):
            out = outData
        print(len(outData.index))
        self._finalModel()
        outTestFeature = out.iloc[0:, featureStartIndex: featureEndIndex]
        test = out.iloc[0:, targetIndex:targetIndex + 1].values.ravel()
        pred = self.finalModel.predict(outTestFeature)
        print(f'{self.finalModel}')
        self.outPred = pred
        self.outObserve = test
        outPredict = self.calculateTestResultMetric(test, pred)
        # outPredDetails = pd.DataFrame([self.outObserve, self.outPred, np.abs(self.outObserve - self.outPred)],
        #                               index=['observed', 'predicted', 'error']).T
        outPredDetails = pd.DataFrame(
            [self.outObserve, self.outPred, np.abs(self.outObserve - self.outPred) / self.outObserve],
            index=['observed', 'predicted', 'error']).T
        outPredResult = pd.DataFrame.from_dict(outPredict, orient='index')
        savePath = f'{self.saveFileFolderPath}//out_obverseVSPredict_{seed}_{self.modelName}_{additiveNameChar}.csv'
        print(savePath)
        outPredDetails.to_csv(
            self.saveFileFolderPath + '/' + 'out_obverseVSPredict' + str(
                seed) + self.modelName + additiveNameChar + '.csv')
        outPredResult.to_csv(
            self.saveFileFolderPath + '/' + 'out_' + str(seed) + self.modelName + additiveNameChar + '.csv')

    def calculateTestResultMetric(self, testData, predData) -> list:
        print(f'testData{testData}')
        print(f'predData{predData}')
        self.predictError = np.abs(predData - testData)
        self.predictPercentError = np.abs(predData - testData) / testData
        testResult = {}
        for score in self.needScoreList:
            score = score.lower()
            if score == 'rmse':
                test_rmse = np.sqrt(mean_squared_error(testData, predData))
                testResult[score] = test_rmse
            if score == 'mae':
                test_mae = mean_absolute_error(testData, predData)
                testResult[score] = test_mae
            if score == 'madae':
                test_madae = median_absolute_error(testData, predData)
                testResult[score] = test_madae
            if score == 'r2':
                test_r2 = r2_score(testData, predData)
                testResult[score] = test_r2
            if score == 'percent_madae':
                test_per_madae = np.median(self.predictPercentError)
                testResult[score] = test_per_madae
            if score == 'percent_mae':
                test_per_mae = np.average(self.predictPercentError)
                testResult[score] = test_per_mae
        # print(testResult['rmse'])
        # print(testResult['r2'])
        return testResult

    def saveFile(self, additiveNameChar=None, verbose=None, timeChar=False):
        """

        :param additiveNameChar: output CSV file name custom char. You can add custom characters to the output file name
        :param verbose: If verbose true , you can get multiple CSV files containing the predicted results of each point
                        in the test set to plotResult.py figure.
        :return:
        """
        additiveChar = ""
        if additiveNameChar:
            additiveChar = additiveNameChar
        else:
            additiveChar = ""
        if timeChar:
            additiveChar += time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fileName = 'seedRange' + '_' + self.seedRangeChar + '_model_' + self.modelName + '_testRadio_' + \
                   str(self.testRatio) + '_' + str(self.gridSearch_Metric) + '_' + additiveChar
        self.printDf = pd.DataFrame(self.resultList)
        self.printDf.set_index('Trial', drop=True, inplace=True)
        self.printDf.to_csv(self.saveFileFolderPath + '/' + fileName + '.csv')
        verboseResult = self.saveFileFolderPath + '/' + fileName + '_verbose' + '.csv'
        self.verboseResult.to_csv(verboseResult)
        df = pd.read_csv(verboseResult, index_col=0)
        df.groupby('params').mean().to_csv(verboseResult)
        if verbose:
            obverseVSPredict = pd.DataFrame([self.y_test, self.y_predict, np.abs(self.y_test - self.y_predict)],
                                            index=['observed', 'predicted', 'error']).T
            obverseVSPredict.to_csv(
                self.saveFileFolderPath + '/' + 'obverseVSPredict' + str(self.seed) + self.modelName +
                additiveChar + '.csv')

    def saveOutFile(self, additiveNameChar=None):
        if self.outPred is not None:
            outPred = pd.DataFrame([self.outObserve, self.outPred, np.abs(self.outObserve - self.outPred)],
                                   index=['observed', 'predicted', 'error']).T
            if additiveNameChar:
                outPred.to_csv(
                    self.saveFileFolderPath + '/' + 'out_obverseVSPredict' + str(self.seed) + self.modelName +
                    additiveNameChar + '.csv')
            else:
                outPred.to_csv(
                    self.saveFileFolderPath + '/' + 'out_obverseVSPredict' + str(self.seed) + self.modelName + '.csv')

    def saveModel(self, filePath):
        if not self.finalModel:
            self._finalModel()
        else:
            None
        with open(filePath, 'wb') as f:
            pkl.dump(self.finalModel, f)

    def useModel(self, filePath, X):
        with open(filePath, 'rb') as pkl_file:
            data = pkl.load(pkl_file)
            y_pred = data.predict(X)
        return y_pred
