import pandas as pd
from matplotlib import pyplot as plt


def readMgfCSV():
    df = pd.read_csv('final.csv')
    ax = plt.subplot()
    df1 = df.loc[df['CHARGE'] == '1+\n']
    ax.scatter(df['PEPMASS'], df['CCS'],s=2,c='r')
    ax.scatter(df1['PEPMASS'], df1['CCS'], s=2,c='b')
    plt.show()
    print()


if __name__ == '__main__':
    readMgfCSV()
