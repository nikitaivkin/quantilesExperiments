#import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt

def plotExp1(resFilePath):
    res= {} 
    for line in open(resFilePath).read().splitlines()[1:]:  # skipping first line/header
        [stream, algo, mode, space, meanError, stdError] = line.split()
        stream = os.path.basename(stream) 
        stream = stream[1:stream.find(".")] 
        if stream not in res:
            res[stream] = {}
        if mode == "NoneNoneNone":
            mode = ""
        if algo + mode not in res[stream]:
            res[stream][algo+mode] = []
        res[stream][algo+mode].append([int(space), float(meanError), float(stdError)])
    
    for stream in res.keys():
        fig, ax = plt.subplots()
        lines = [] 
        algos = ['lwyc', 'kll0000', 'kll1000', 'kll0100', 'kll0010', 'kll0001']
        for algo in algos:
            data = np.array(res[stream][algo])
            line, = ax.plot(range(len(data[:,0])),data[:,1]/10**6, linewidth=2, label= algo)
            lines.append(line)
        
        ax.legend(loc='upper right', fontsize=14)
        plt.yticks(fontsize=16)
        plt.xticks(range(8),[128,256,512,1024,2048,4096,8192,16384],fontsize=16)
        ax.set_title(stream) 
        #ax.set_ylim(0, 0.1)
        ax.set_xlim(0, 7)
        ax.set_xlabel("Sketch size", fontsize=18)
        ax.set_ylabel('Error', fontsize=18)
        ax.grid(linestyle='-', linewidth=0.5)
        plt.tight_layout()
        #plt.savefig('tricks1.png')
        plt.show()

#
#    plt.errorbar(xrange(len(error_mean)), error_mean/1000000, error_std/1000000, linestyle='None', marker='^')
#    plt.xticks(xrange(len(error_mean)), mode,  rotation='vertical')
#    plt.xlim(-1,14)
#    plt.xlabel("algorithm mode, 0 - off, 1 - on (greedy memory, lazy compactions, sampling, one coin flip, shifting trick, one pair compaction (random - 1, non random -2)) ", fontsize=16)
#    plt.ylabel("Mean relative error +/- standard deviation",fontsize=24)
#    plt.show()
#    #print (df_mean.filter(like='random', axis=0))
#    #print(df_mean.loc[df_mean['dsType'] == 'random'])

if __name__ == "__main__":
    plotExp1('6exp1.out')

