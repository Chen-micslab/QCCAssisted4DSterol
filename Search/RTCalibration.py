import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
import pandas as pd
import numpy as np
from myplot.plotUtils import *


def RTCalibrationByRTQC(StandardRT: list, RTQC: list, predRT: list = [], ax:object=None,index:int=None):
    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(StandardRT, RTQC, frac=1, it=1)
    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    trendLinex = [i for i in range(0, 2500, 1)]
    trendLiney = f(trendLinex)

    l1, = ax.plot(StandardRT, RTQC, 'ob', label='Origin RT',c='black', markersize=5)
    # validation = [624, 546, 450, 504, 522, 276, 528, 528, 600, 1224, 1308, 1266, 1266, 642, 654, 402, 666, 270, 60, 66,
    #               582, 582]
    # print(f(validation))
    l2, = ax.plot(StandardRT, f(StandardRT), 'or', label='After Calibration RT', c='red', markersize=5)
    # l2, = ax.plot(lowess_x, lowess_y, 'or', label='After Calibration RT',c='red')
    l3, = ax.plot(trendLinex, trendLiney, '-k', label='Calibration Line', markersize=5)
    # l4 = ax.plot(np.arange(0,1500,0.01),np.arange(0,1500,0.01), '-k',label=f'y=x',c='r')
    legend_font = {
        'family': 'Arial',  # 字体族
        'size': 8,  # 字体大小
    }
    ax.legend(handles=[l1, l2, l3],prop=legend_font)
    ax.set_xlabel('Standard RT (s)', fontdict=fontdict_lable_generate(8), labelpad=1)
    ax.set_ylabel('Calibration RT (s)', fontdict=fontdict_lable_generate(8), labelpad=1)
    ax.set_title(f'RTQC{index+1} calibration curve',fontdict=fontdict_lable_generate(8))
    plt.setp(ax.get_xticklabels(), **fontdict_lable_generate(8))
    plt.setp(ax.get_yticklabels(), **fontdict_lable_generate(8))
    result = []
    for x, y in zip(f(predRT),predRT):
        if np.isnan(x):
            result.append(y)
        else:
            result.append(x)
    return result


if __name__ == '__main__':
    database = [66, 270, 522, 546, 624, 666, 654, 1308]
    QC1 = [72, 274, 528, 552, 618, 660, 648, 1302]
    QC2 = [72, 276, 534, 554, 620, 658, 648, 1302]
    QC3 = [72, 281, 534, 553, 622, 661, 648, 1308]
    fig, axs = plt.subplots(1, 3, figsize=(6.5/2.54*3,6.5/2.54),layout='constrained')
    databasePredRT = '../Search/database/STDataBase.csv'
    df = pd.read_csv(databasePredRT, index_col=0)
    databaseRTList = df['RT'].values.astype(float)
    df.index.name = 'LMID'
    df.columns = ['MZ', 'diagIonsInChian', 'diagIonsInRing', 'MS2', 'RT', 'CCS','Standard']
    for i, qc in enumerate([QC1, QC2, QC3]):
        newRT = RTCalibrationByRTQC(database, qc, databaseRTList, axs[i], index=i)
        df.loc[:, 'RT'] = newRT
        df.to_csv(f'../Search/database/STDatabaseQC{i + 1}.csv', index=True)
    plt.show()
