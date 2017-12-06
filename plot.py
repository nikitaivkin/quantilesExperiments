import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def plot(space, dataset): 
    df = pd.read_csv('results/results.csv', names=['dataset', 'algo', 'space', 'mode', 'c', 'repetition','error', 'estVar'], dtype = {'mode':np.str})
    df['dsType'] = pd.Series(df['dataset'], index=df.index)
    rep = {"./datasets/r6.npy": "random","./datasets/zi6.npy": "zoomin","./datasets/zo6.npy": "zoomout","./datasets/s6.npy": "sorted" }
    df['dsType'] = df['dsType'].replace(rep)
    del df["dataset"]
    del df["c"]
    del df["repetition"]
    del df["estVar"]
    df = df.loc[df['dsType'] == 'random'] 
    df = df.loc[df['space'] == 512] 
    df = df.loc[df['algo'] ==df['algo'][0]] 
    del df['dsType']
    del df['space']
    del df['algo']
    #print(df['mode'].drop_duplicates())
    #print(df['dataset'].drop_duplicates())
    #print(df)
    #grouped = df.groupby(['dataset', 'algo','space','mode','c'])
    #for name, group in grouped:
    #    print(name)
    #    print(group)
    df_mean = df.groupby(['mode']).mean()
    df_std = df.groupby(['mode']).std()
    #print (df_mean)
    #print (df_std)
    df_mean['std'] = df_std['error']
    df_mean = df_mean.reset_index(level=0)
    print (df_mean)
    error_mean =  np.array(df_mean['error'])
    error_std = np.array(df_mean['std'])
    mode = np.array(df_mean['mode'])
    print(error_mean)
    print(error_std)
    print(mode)
    plt.errorbar(xrange(len(error_mean)), error_mean/1000000, error_std/1000000, linestyle='None', marker='^')
    plt.xticks(xrange(len(error_mean)), mode,  rotation='vertical')
    plt.xlim(-1,14)
    plt.xlabel("algorithm mode, 0 - off, 1 - on (greedy memory, lazy compactions, sampling, one coin flip, shifting trick, one pair compaction (random - 1, non random -2)) ", fontsize=16)
    plt.ylabel("Mean relative error +/- standard deviation",fontsize=24)
    plt.show()
    #print (df_mean.filter(like='random', axis=0))
    #print(df_mean.loc[df_mean['dsType'] == 'random'])

if __name__ == "__main__":
    plot(512, 'random')

