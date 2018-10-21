import kll
import lwyc
from data import *
from functools import partial
from multiprocessing import Pool
import numpy as np
import sys

def runAlgo(algo, params, stream, **kvargs): 
    ds = algo(s=params["s"], c=params["c"], mode=params["mode"])      
    for  item in stream:
        ds.update(item)
    return ds.evalMaxError(stream) 

def runAlgoNreps(algo, params, stream, repsN, threads=1):
    if threads == 1: 
        errors = [runAlgo(algo, params, stream) for _ in range(repsN)] 
    else:
        pool = Pool(processes=threads)
        runOneModePartial = partial(runAlgo,  params=params, stream=stream)
        errors = pool.map(runOneModePartial, [algo]*repsN)
        pool.close(); pool.join()
    return [np.mean(errors), np.std(errors)] 

def runManySettings(algo, params, streams, repsN, threads=1):
    spaces = params["s"];
    modes = params["mode"]
    print("dataset|algo|mode|sketchsize|error|errorStd")
    for s_name, s_data in streams.iteritems():
        for space in spaces: 
            for mode in modes: 
                params["s"] = space
                params["mode"] = mode
                result = runAlgoNreps(algo, params, s_data, repsN, threads=1)
                print(s_name + "\t" + str(algo)[-4:]+" \t" + "".join(map(str,mode)) + "\t" + str(space) + "\t"+ 
                      str("{:10.1f}".format(result[0]))+ "\t"+ str("{:10.1f}".format(result[1])))
    return 0


def exp1(streamLen, repsN, threadsN):
    # loading all streams of length streamLen
    streamNames = ["random", "sorted", "caida", "wiki", "wiki_s"]
    streams = dict([(s, Data.load("streams/" + str(streamLen) + s + ".npy")) for s in streamNames])  
    
    # running experiments for KLL for all modes and spaces  
    algo = getattr(kll,"KLL") 
    params = {"s": [128, 256, 512, 1024, 2048, 4096, 8192, 16384], 
              "c": 2./3.,
              "mode": [(0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0),
                       (0,0,1,0),(1,0,1,0),(0,1,1,0),(1,1,1,0),
                       (0,0,0,1),(1,0,0,1),(0,1,0,1),(1,1,0,1),
                       (0,0,1,1),(1,0,1,1),(0,1,1,1),(1,1,1,1)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
   
    # running experiments for LWYC for all modes and spaces  
    algo = getattr(lwyc,"LWYC") 
    params = {"s":[128, 256, 512, 1024, 2048, 4096, 8192, 16384], 
              "c": None,
              "mode": [(None,None,None)]} 
    runManySettings(algo, params, streams, repsN, threadsN)


if __name__ == "__main__":
    exp1(4, 10, 2)
