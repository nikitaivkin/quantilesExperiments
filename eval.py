import kll, lwyc
from data import *
import numpy as np
from functools import partial
from multiprocessing import Pool
import argparse, sys

def evalMaxError(streamFilePath, estRanks, itemType="str"):
    trueRanks = np.zeros(len(estRanks) + 1)
    with open(streamFilePath,'rU') as f:
        for item in f:
            if itemType=="int":
                trueRanks[np.searchsorted(estRanks[:,0], int (item), side="right")] += 1 
            else:
                trueRanks[np.searchsorted(estRanks[:,0], item, side="right")] += 1 
    trueRanks = np.cumsum(trueRanks[:-1])
    estRanks = np.array(estRanks[:,1],dtype=np.int32)
    maxError = max([abs(i-j) for i,j in zip(estRanks, trueRanks)])
    return maxError    

def runAlgo(algo, params, streamFilePath, itemType="str", **kvargs): 
    ds = algo(s=params["s"], c=params["c"], mode=params["mode"])      
    with open(streamFilePath) as f:
        for item in f:
            if itemType == "int": 
                ds.update(int(item))
            else:
                ds.update(item)
    return evalMaxError(streamFilePath, np.array(ds.ranks()), itemType) 

def runAlgoNreps(algo, params, streamFilePath, itemType, repsN, threads=1):
    if threads == 1: 
        errors = [runAlgo(algo, params, streamFilePath, itemType) for _ in range(repsN)] 
    else:
        pool = Pool(processes=threads)
        runOneModePartial = partial(runAlgo,  params=params, streamFilePath=streamFilePath, itemType=itemType)
        errors = pool.map(runOneModePartial, [algo]*repsN)
        pool.close(); pool.join()
    return [np.mean(errors), np.std(errors)] 

def runManySettings(algo, params, streams, repsN, threads=1, printC=0):
    spaces = params["s"];
    modes = params["mode"]
    algoname = "kll" if "KLL" in str(algo) else "lwyc" 
    for [streamFilePath, itemType] in streams:
        for space in spaces: 
            for mode in modes: 
                params["s"] = space
                params["mode"] = mode
                result = runAlgoNreps(algo, params, streamFilePath, itemType, repsN, threads)
                print(streamFilePath + "\t" + algoname +" \t" + "".join(map(str,mode)) + "\t" + str(space) + "\t"+ 
                       (str(params['c']) + "\t")*printC + 
                       str("{:10.1f}".format(result[0]))+ "\t"+ str("{:10.1f}".format(result[1])))
    return 0

def loadStreams(streamLen, streamsPath, stream):
    # loading all streams of length streamLen
    streams = ["random", "sorted", "brownian", "trending", "caida", "wiki", "wiki_s"]
    types = ["int", "int", "int", "int", "str", "str", "str"]
    if stream!= 'NA':
        types = [types[streams.index(stream)]]
        streams = [stream]
    streams = [streamsPath + str(streamLen)+ i + ".csv" for i in streams]
    return zip(streams, types) 

def exp1(streamLen, streamsPath, stream, repsN, threadsN):
    streams = loadStreams(streamLen, streamsPath, stream) 
    
    # header for the output
    print("dataset|algo|mode|sketchsize|error|errorStd") 
    
    # running experiments for LWYC for all modes and spaces  
    algo = getattr(lwyc,"LWYC") 
    params = {"s":[128, 256, 512, 1024, 2048, 4096, 8192, 16384], 
              "c": None,"mode": [(None,None,None)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
    
    # running experiments for KLL for all modes and spaces  
    algo = getattr(kll,"KLL") 
    params = {"s": [128, 256, 512, 1024, 2048, 4096, 8192, 16384], "c": 2./3.,
              "mode": [(0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0), (0,0,1,0),(1,0,1,0),
                       (0,1,1,0),(1,1,1,0),(0,0,0,1),(1,0,0,1),(0,1,0,1),(1,1,0,1),
                       (0,0,1,1),(1,0,1,1),(0,1,1,1),(1,1,1,1)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
   
def exp2(streamLen, streamsPath, stream, repsN, threadsN):
    streams = loadStreams(streamLen, streamsPath, stream) 
    
    # header for the output
    print("dataset|algo|mode|sketchsize|c|error|errorStd") 
   
    # running experiments for KLL for all modes and spaces  
    algo = getattr(kll,"KLL") 
    crange = np.arange(10,95,5)/100.
    for c in crange: 
        params = {"s": [512], "c": c,
                  "mode": [(0,0,0,0)]} 
        runManySettings(algo, params, streams, repsN, threadsN, printC=1)
    
    for c in crange: 
        params = {"s": [512], "c": c,
                  "mode": [(1,0,0,0)]} 
        runManySettings(algo, params, streams, repsN, threadsN, printC=1)
  
def exp3(streamLens=[4,8],streamsPath="streams/", stream='random', repsN=1, threadsN=1):
    streams = [] 
    for streamLen in range(streamLens[0],streamLens[1]+1):
        streams.extend(loadStreams(streamLen, streamsPath, stream)) 
    # header for the output
    print("dataset|algo|mode|sketchsize|error|errorStd") 
    
    # running experiments for LWYC for all modes and spaces  
    algo = getattr(lwyc,"LWYC") 
    params = {"s":[256,512,1024], 
              "c": None,"mode": [(None,None,None)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
    
    # running experiments for KLL for all modes and spaces  
    algo = getattr(kll,"KLL") 
    params = {"s": [256, 512, 1024], "c": 2./3.,
              "mode": [(0,0,0,0),(1,1,0,1),(1,1,1,1)]}
    runManySettings(algo, params, streams, repsN, threadsN)
 
def exp4(streamLen, streamsPath, repsN, threadsN):
    streams = [] 
    # loading all streams of length streamLen
    ps = [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    types = ["int"] * len(streams)
    streams = [streamsPath + str(streamLen)+ "trending" + str(i) +  ".csv" for i in ps]
    streams = zip(streams, types) 
    # header for the output
    print("dataset|algo|mode|sketchsize|error|errorStd") 
    
    # running experiments for LWYC for all modes and spaces  
    algo = getattr(lwyc,"LWYC") 
    params = {"s":[256], 
              "c": None,"mode": [(None,None,None)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
    
    print("dataset|algo|mode|sketchsize|error|errorStd") 
    # running experiments for KLL for all modes and spaces  
    algo = getattr(kll,"KLL") 
    params = {"s": [ 256], "c": 2./3.,
              "mode": [(0,0,0,0),(1,0,0,0),(1,1,0,1),(1,1,1,1)]} 
    runManySettings(algo, params, streams, repsN, threadsN)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, default="exp1", 
                        help='experiment to run: "exp1", "exp2", ...')
    parser.add_argument('-l', type=int, default=4, 
                        help='length of the stream (set x to get 10^x)')
    parser.add_argument('-r', type=int, default=10, 
                        help='number of repetitions for each run')
    parser.add_argument('-t', type=int, default=1, 
                        help='number of threads to use')
    parser.add_argument('-p', type=str, default="streams/", 
                        help='path to all streams')
    parser.add_argument('-s', type=str, default="NA", 
                        help='dataset')
    args = parser.parse_args()
    
    if  args.a  == 'exp1':
        exp1(streamLen=args.l, streamsPath=args.p, stream=args.s, repsN=args.r , threadsN=args.t)
    elif args.a  == 'exp2':
        exp2(streamLen=args.l, streamsPath=args.p, stream=args.s, repsN=args.r , threadsN=args.t)
    elif args.a  == 'exp3':
        exp3(streamLens=[args.l,args.l+1], streamsPath=args.p, repsN=args.r , threadsN=args.t)
    elif args.a  == 'exp4':
        exp4(streamLen=args.l, streamsPath=args.p, repsN=args.r , threadsN=args.t)

