"""
A machine learning regression module program for predicting tabular data based scikit-learn library.
==================================
by: Sun Jian
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
import pickle as pkl
from typing import Union
from sklearn.feature_selection import SelectFromModel
import os


class TabularRegressionML:
    def __init__(self, reaction_data, save_file_path: str, seed_start: int, seed_stop: int, seed_step: int,
                 test_ratio: float = 0.2, cv_number=10, score=[], splitRule=None):
        self.seed_range = np.arange(seed_start, seed_stop, seed_step)
        self.seed_range_char = f'{seed_start}_{seed_stop}_{seed_step}'
        self.seed: Union[None, int] = None
        self.test_ratio: float = test_ratio
        self.print_df = pd.DataFrame()
        if score is None:
            self.need_score_list = ['RMSE', 'R2', 'MAE']
        elif isinstance(score, list):
            self.need_score_list = score
        else:
            raise Exception(f'wrong score input need list or none')
        self.save_file_folder_path = save_file_path
        self.cv_number = cv_number
        self.feature_target = pd.DataFrame
        self.feature = pd.DataFrame
        self.target = pd.DataFrame
        self.cv = KFold(n_splits=5, shuffle=False, random_state=None)
        self.test_result = {}
        self.result = {}
        self.result_list = []
        self.grid_search_metric: str = 'neg_root_mean_squared_error'
        self.model_number: Union[None, float] = None
        self.model_name: Union[None, float] = None
        self.grid: dict = {}
        self.predict_error = None
        self.y_pred: np.ndarray = np.array([])
        self.y_test: np.ndarray = np.array([])
        self.X_test: np.ndarray = np.array([])
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])
        self.best_estimator: object = None
        self.baseline: Union[None, float] = None
        self.out_pred = None
        self.out_observe = None
        self.split_rule = splitRule
        self.final_model = None
        self.predict_percent_error = 0
        self.verbose_result = pd.DataFrame(columns=['Trial', 'params', 'mean_score'])
        self.model_number: Union[None, int] = None
        assert reaction_data is not None, 'Need data'
        if isinstance(reaction_data, pd.DataFrame):
            self.reactionData = reaction_data
        elif isinstance(reaction_data, str):
            self.reactionData = pd.read_csv(reaction_data, encoding="unicode_escape", low_memory=False)
        else:
            raise Exception(f'Check Reaction Datatype')

    def set_seed_range(self, seed_start: int, seed_stop: int, seed_step: int):
        self.seed_range = np.arange(seed_start, seed_stop, seed_step)

    def analyze_tabular_data(self, featureIndexList: list, targetIndex: int):
        print('-----------------------tabular data----------------------------\n')
        self.feature_target = pd.DataFrame(index=self.reactionData.index)
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
                        pass
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
        self.feature_target = pd.concat([self.target, self.feature], axis=1)
        print('Analyze Reaction Data Success\n')
        print(f'Feature Number is {featureNumber}\n')
        print(f'Feature is {self.feature.columns}\n')
        print('Predict Target Name: {targetName}\n'.format(
            targetName=self.reactionData.iloc[:, targetIndex:targetIndex + 1].columns.values[0]))
        print(f'Get Data. Now Please Set The CV Parameters\n')
        print(f'total data number is {len(self.target)}\n')
        print('-----------------------analyzeReactionDataEnd--------------------------\n')
        return self

    def model_selection_input(self):
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
        assert function_Number in [str(x) for x in range(0, 20, 1)], 'Number Wrong'
        self.model_number = function_Number

    def set_grid(self, gridParameter: list):
        self.grid = gridParameter

    def _chooseModel(self, seed):
        rng = np.random.RandomState(seed)
        if 1 == int(self.model_number):
            self.model_name = 'RandomForest'
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
        elif 2 == int(self.model_number):
            self.model_name = 'XgBoost'
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
        elif 3 == int(self.model_number):
            self.model_name = 'AdaBoost'
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
        elif 4 == int(self.model_number):
            self.model_name = 'GBDT'
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
        elif 5 == int(self.model_number):
            self.model_name = 'CatBoost'
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
        elif 6 == int(self.model_number):
            self.model_name = 'LightGBM'
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
        elif 7 == int(self.model_number):
            pca = PCA(n_components=0.85, random_state=seed)
            self.model_name = 'SVM'
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
        elif 8 == int(self.model_number):
            self.model_name = 'Lasso'
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
        elif 9 == int(self.model_number):
            pca = PCA(n_components=0.99, random_state=seed)
            self.model_name = 'Ridge'
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
        elif 10 == int(self.model_number):
            pca = PCA(n_components=0.99, random_state=seed)
            self.model_name = 'ElasticNet'
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
        elif 11 == int(self.model_number):
            self.model_name = 'KNN'
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
        elif 0 == int(self.model_number):
            self.model_name = 'debugRandomForest'
            self.grid = [{
                'regressor__n_estimators': [2]
            }]
            model = Pipeline(
                [
                    ("regressor", RandomForestRegressor(random_state=seed))
                ]
            )
            return model
        elif 12 == int(self.model_number):
            pca = PCA(n_components=0.8, random_state=seed)
            self.model_name = 'PCA-SVR'
            model = Pipeline(
                [("standard", StandardScaler()),
                 ("PCA", pca),
                 ("regressor", SVR())
                 ]
            )
            return model

    def start_grid_search(self, seed, gridSearch_Metric='neg_root_mean_squared_error', jobs: int = 7) -> None:
        self.seed = seed
        self.grid_search_metric = gridSearch_Metric
        rng = np.random.RandomState(seed)
        print(f'now seed is {self.seed}')

        X_train, X_test, y_train, y_test = train_test_split(self.feature, self.target.values.ravel(),
                                                            test_size=self.test_ratio, random_state=rng)
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        estimator = self._chooseModel(self.seed)
        gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                          scoring=gridSearch_Metric, cv=self.cv, verbose=5, n_jobs=jobs)
        gs.fit(X_train, y_train)
        self.baseline = self.y_test.mean()
        self.best_estimator = gs.best_estimator_
        self.y_pred = gs.best_estimator_.predict(X_test)
        self._generate_verbose_trail_result(self.seed, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
        self.result = self.calculate_test_result_metric(self.y_test, self.y_pred)
        self.result['Trial'] = self.seed
        self.result['best_parameter'] = gs.best_params_
        if gs.best_score_ < 0:
            self.result['best_train_score'] = -gs.best_score_
        else:
            self.result['best_train_score'] = gs.best_score_
        print(self.result)
        self.print_df = pd.concat([self.print_df, pd.DataFrame.from_dict(self.result, orient='index').T], axis=0)
        print(f'Grid_search:{gs.best_params_}')
        return None

    def ML_GridSearchLOO(self, seed, gridSearch_Metric='neg_root_mean_squared_error') -> None:
        self.seed = seed
        self.grid_search_metric = gridSearch_Metric
        rng = np.random.RandomState(seed)
        print(f'now seed is {self.seed}')
        feartureDrop = self.feature.drop(index=self.feature.index[seed])
        targetDrop = self.target.drop(index=self.feature.index[seed])
        X_train, X_test, y_train, y_test = train_test_split(feartureDrop, targetDrop.values.ravel(),
                                                            test_size=self.test_ratio, random_state=rng)
        self.X_test = X_test
        self.y_test = y_test
        estimator = self._chooseModel(self.seed)
        gs = GridSearchCV(estimator=estimator, param_grid=self.grid,
                          scoring=gridSearch_Metric, cv=self.cv, verbose=5, n_jobs=7)
        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)
        self._generate_verbose_trail_result(self.seed, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
        print()
        self.y_pred = y_pred
        self.result = self.calculate_test_result_metric(self.y_test, self.y_pred)
        self.result['Trial'] = self.seed
        self.result['best_parameter'] = gs.best_params_
        self.result['best_train_score'] = -gs.best_score_
        self.result_list.append(self.result)
        L00pred = gs.best_estimator_.predict(self.feature.iloc[[seed], :])
        self.result['XLOO'] = self.target.values[seed][0]
        self.result['YLOO'] = L00pred[0]
        self.result['LOO_Relative_Error'] = (abs(self.result['XLOO'] - self.result['YLOO']) / self.result['XLOO'])
        print(f'LOO pred {L00pred}')
        print(f'Grid_search:{gs.best_params_}')
        return None

    def _generate_verbose_trail_result(self, trial, params: list, meantest: np.dtype):
        oneVerboseResult = pd.DataFrame()
        oneVerboseResult['Trial'] = [str(trial)] * len(params)
        oneVerboseResult['params'] = params
        oneVerboseResult['mean_score'] = meantest
        self.verbose_result = pd.concat([self.verbose_result, oneVerboseResult], axis=0)

    def vs_baseline(self):
        print('baseline\n')
        baseLineList = self.baseline.values[0] * np.ones(self.y_test.shape)
        baseLine_result = self.calculate_test_result_metric(baseLineList, self.y_test)
        self.result['base_rmse'] = baseLine_result['rmse']
        self.result['base_r2'] = baseLine_result['r2']

    def LOO_Score(self, gridSearch_Metric='neg_root_mean_squared_error', verbose: bool = False, addChar: str = ''):
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
            self._generate_verbose_trail_result(i, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
            self.result['best_parameter'] = gs.best_params_
            self.result['best_train_score'] = -gs.best_score_
            self.result_list.append(self.result)
            a = np.abs(y_pred - y_test) / y_test
            print(f'ytest{y_test} y_pred{y_pred} relative error{a[0]:.2%}')
            MREscore.append(a[0])
            predict.append(y_pred[0])
        if verbose:
            predDetails = pd.DataFrame(
                [self.target.values.ravel(), predict, MREscore],
                index=['observed', 'predicted', 'error']).T
            predDetails.to_csv(f'{self.save_file_folder_path}/LOO.csv')
        fileName = 'seedRange' + '_' + self.seed_range_char + '_model_' + self.model_name + '_testRadio_' + \
                   str(self.test_ratio) + '_' + str(self.grid_search_metric) + addChar
        self.print_df = pd.DataFrame(self.result_list)
        verboseResult = self.save_file_folder_path + '/' + fileName + '_verbose' + '.csv'
        self.verbose_result.to_csv(verboseResult)
        df = pd.read_csv(verboseResult, index_col=0)
        df.groupby('params').mean().to_csv(verboseResult)
        print(f'{np.mean(MREscore):.2%}')
        print(f'{np.median(MREscore):.2%}')
        print(f'{gs.best_estimator_}')

    def Loo_score_with_assisted(self, assitedPd, gridSearch_Metric='neg_root_mean_squared_error', verbose: bool = False,
                                addChar: str = ''):
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
            self._generate_verbose_trail_result(i, gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
            self.result['best_parameter'] = gs.best_params_
            self.result['best_train_score'] = -gs.best_score_
            self.result_list.append(self.result)
            a = np.abs(y_pred - y_test) / y_test
            print(f'ytest{y_test} y_pred{y_pred} relative error{a[0]:.2%}')
            MREscore.append(a[0])
            predict.append(y_pred[0])
        if verbose:
            predDetails = pd.DataFrame(
                [self.target.values.ravel(), predict, MREscore],
                index=['observed', 'predicted', 'error']).T
            predDetails.to_csv(f'{self.save_file_folder_path}/LOOAssist.csv')
        fileName = 'seedRange' + '_' + self.seed_range_char + '_model_' + self.model_name + '_testRadio_' + \
                   str(self.test_ratio) + '_' + str(self.grid_search_metric) + '_' + addChar
        self.print_df = pd.DataFrame(self.result_list)
        verboseResult = self.save_file_folder_path + '/' + fileName + '_verbose' + '.csv'
        self.verbose_result.to_csv(verboseResult)
        df = pd.read_csv(verboseResult, index_col=0)
        df.groupby('params').mean().to_csv(verboseResult)
        print(f'{np.mean(MREscore):.2%}')
        print(f'{np.median(MREscore):.2%}')
        print(f'{gs.best_estimator_}')

    def train_final_model(self, seed: int = 0):
        self.final_model = self._chooseModel(seed)
        self.final_model.set_params(**self.grid)
        self.final_model.fit(self.feature, self.target.values.ravel())


    def out_sample_predict(self, out_data, target_index, feature_start_index, feature_end_index,
                           additiveNameChar=None):
        print(f'out sample')
        seed = 1
        if isinstance(out_data, str):
            out = pd.read_csv(out_data, encoding="unicode_escape", low_memory=False)
        elif isinstance(out_data, pd.core.frame.DataFrame):
            out = out_data
        print(len(out_data.index))
        self.train_final_model()
        outTestFeature = out.iloc[0:, feature_start_index: feature_end_index]
        test = out.iloc[0:, target_index:target_index + 1].values.ravel()
        pred = self.final_model.predict(outTestFeature)
        print(f'{self.final_model}')
        self.out_pred = pred
        self.out_observe = test
        outPredict = self.calculate_test_result_metric(test, pred)
        # outPredDetails = pd.DataFrame([self.outObserve, self.outPred, np.abs(self.outObserve - self.outPred)],
        #                               index=['observed', 'predicted', 'error']).T
        outPredDetails = pd.DataFrame(
            [self.out_observe, self.out_pred, np.abs(self.out_observe - self.out_pred) / self.out_observe],
            index=['observed', 'predicted', 'error']).T
        outPredResult = pd.DataFrame.from_dict(outPredict, orient='index')
        savePath = f'{self.save_file_folder_path}//out_obverseVSPredict_{seed}_{self.model_name}_{additiveNameChar}.csv'
        print(savePath)
        outPredDetails.to_csv(
            self.save_file_folder_path + '/' + 'out_obverseVSPredict' + str(
                seed) + self.model_name + additiveNameChar + '.csv')
        outPredResult.to_csv(
            self.save_file_folder_path + '/' + 'out_' + str(seed) + self.model_name + additiveNameChar + '.csv')

    def calculate_test_result_metric(self, testData, predData) -> dict:
        print(f'testData{testData}')
        print(f'predData{predData}')
        self.predict_error = np.abs(predData - testData)
        self.predict_percent_error = np.abs(predData - testData) / testData
        testResult = {}
        for score in self.need_score_list:
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
                test_per_madae = np.median(self.predict_percent_error)
                testResult[score] = test_per_madae
            if score == 'percent_mae':
                test_per_mae = np.average(self.predict_percent_error)
                testResult[score] = test_per_mae
        # print(testResult['rmse'])
        # print(testResult['r2'])
        return testResult

    def select_feature_from_model(self):
        feartureDrop = self.feature.drop(index=self.feature.index[self.seed])
        targetDrop = self.target.drop(index=self.feature.index[self.seed])
        estimator = self._chooseModel(self.seed)
        SelectFromModel(estimator).fit(feartureDrop.values, targetDrop.values.ravel())
        return SelectFromModel.get_support()

    def save_result_file(self, additive_name_char: Union[str, None] = None, test_verbose: bool = False,
                         time_char: bool = False) \
            -> None:
        """

        :param time_char:
        :param additive_name_char: output CSV file name custom char. You can add custom characters to the output file name
        :param test_verbose: If verbose true , you can get multiple CSV files containing the predicted results of each point
                        in the test set to plotResult.py figure.
        :return:
        """
        fileName = f'seedRange_{self.seed_range_char}_model_{self.model_name}_testRadio_' \
                   f'{self.test_ratio}_{self.grid_search_metric}'
        if additive_name_char:
            fileName = f'{fileName}_{additive_name_char}'
        if time_char:
            fileName = f'{fileName}_{time_char}'
        verbose_save_path = f'{fileName}_verbose.csv'
        file_save_path = f'{fileName}.csv'
        test_save_path = f'{fileName}_test_verbose.csv'
        file_save_path = os.path.join(self.save_file_folder_path, file_save_path)
        test_save_path = os.path.join(self.save_file_folder_path, test_save_path)
        verbose_save_path = os.path.join(self.save_file_folder_path, verbose_save_path)
        self.print_df.set_index('Trial', drop=True, inplace=True)
        self.print_df.to_csv(file_save_path)
        self.verbose_result.to_csv(verbose_save_path)  # params 作为index 是非哈希 输出在读成 str astype转str不太好
        df = pd.read_csv(verbose_save_path, index_col=0)
        df = df.groupby(['params']).mean().drop('Trial', inplace=False, axis=1)
        df.sort_values(by='mean_score', inplace=True, ascending=False)
        print(f'best parameter \n {df.iloc[0, :]}')
        df.to_csv(verbose_save_path)
        if test_verbose:
            obverseVSPredict = pd.DataFrame([self.y_test, self.y_pred, np.abs(self.y_test - self.y_pred)],
                                            index=['observed', 'predicted', 'abs error']).T
            obverseVSPredict.to_csv(test_save_path)
        self.print_df.drop(self.print_df.index, inplace=True)
        self.verbose_result.drop(self.verbose_result.index, inplace=True)

    def clear_output_file(self):
        self.print_df.drop(self.print_df.index, inplace=True)
        self.verbose_result.drop(self.verbose_result.index, inplace=True)

    # def save_out_file(self, additiveNameChar=None): if self.out_pred is not None: outPred = pd.DataFrame([
    # self.out_observe, self.out_pred, np.abs(self.out_observe - self.out_pred)], index=['observed', 'predicted',
    # 'error']).T if additiveNameChar: outPred.to_csv( self.save_file_folder_path + '/' + 'out_obverseVSPredict' +
    # str(self.seed) + self.model_name + additiveNameChar + '.csv') else: outPred.to_csv( self.save_file_folder_path
    # + '/' + 'out_obverseVSPredict' + str(self.seed) + self.model_name + '.csv')

    def save_model(self, filePath):
        if not self.final_model:
            self.train_final_model()
        else:
            None
        with open(filePath, 'wb') as f:
            pkl.dump(self.final_model, f)

    def use_model(self, filePath, X):
        with open(filePath, 'rb') as pkl_file:
            data = pkl.load(pkl_file)
            y_pred = data.predict(X)
        return y_pred
