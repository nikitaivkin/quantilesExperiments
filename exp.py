import numpy as np
from random import random
import logging
from math import sqrt, log
from algos import *
# import algos
from data import Data
from multiprocessing import Pool
from functools import partial

def doOneRun(setting):
    dataPath = setting[0]
    data = Data.load(dataPath)
    sketchName = setting[1]
    space = int(setting[2])
    dataN = 10 ** int(setting[0][-5])
    algoMode = map(int, list(setting[3]))
    cParam = float(setting[4])
    # sketch = getattr(algos, sketchName)(s=space,c=cParam, mode=algoMode, n=dataN)
    sketch = globals()[sketchName](s=space,c=cParam, mode=algoMode, n=dataN)
    
    rep = int(setting[5])
    for item_i, item in enumerate(data):
        sketch.update(item)
    estRanks = sketch.ranks()
    nums, estRanks = zip(*estRanks)
    realRanks = Data.getQuantiles(data, nums)
    settingNew = setting[:]
    settingNew.append(str(np.max(np.abs(np.array(realRanks) - np.array(estRanks)))))
    settingNew.append(str(np.sqrt(sketch.cumVar)))
    return ", ".join(settingNew)


def runExpWithPool(start, end, nProcesses):
    queue = readSettingQueue("queue.csv")
    queue = queue[start:end]
    pool = Pool(processes=nProcesses)
    # method_to_call =globals()["KLL"]

    results = pool.map(doOneRun, queue)
    pool.close()
    pool.join()
    resFile = open("results.csv", "w")
    for result in results:
        resFile.write(result)
        resFile.write("\n")

    resFile.close()

def genQueue(datasets, algos, srange, modes, crange,repsNum, path):
    f = open(path, "w")
    c = 2./3.
    for dataset in datasets:
        for algo in algos:
            for space in srange:
                for mode in modes:
                    # for c in crange:
                    for rep in range(repsNum):
                        f.write(" ".join([dataset,algo,str(space),mode, str(c), str(rep)]) + "\n")
                        # if algo == 'CormodeRandom' or algo == 'MRL': break;
                    if algo == 'CormodeRandom' or algo == 'MRL': break;




def readSettingQueue(path):
    params = []
    for line in tuple(open(path, 'r')):
        params.append(line.rstrip().split())
    return params


if __name__ == '__main__':

    
    
    datasets = ["./datasets/r6.npy", "./datasets/s6.npy", "./datasets/zi6.npy", "./datasets/zo6.npy"]
    algos = ['KLL', 'MRL', 'CormodeRandom']
    srange = 2**np.array(range(9,13))
    modes = ["00000","10000","20000","21000","21100","21110","21111"]
    crange = np.arange(0.1, 0.5, 0.05)
    repsNum = 50
    path = "./queue.csv"
    genQueue(datasets, algos, srange, modes, crange, repsNum, path)
    runExpWithPool(1,10,2)

    # runAllExp()
    # runExpWithPool
    # runExp(0, 10)
