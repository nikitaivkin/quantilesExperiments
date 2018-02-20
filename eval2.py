from klls import *
from lwyc import *
from lwycLazy import *
from data import *
import numpy as np
import sys
import socket
#import matplotlib.pyplot as plt

CONVERSION_NEEDED = 0
ALL_UNIQUE= 0
READING_FROM_FILE = 0

def runManyReps(nProcesses, reps, partialF):
    pool = Pool(processes=nProcesses)
    res = pool.map(partialF, range(reps))
    pool.close()
    pool.join()
    return res 

def runKLL(_=0, mode=[0,0,0,0,0], space=128, stream_=[], c=2./3.):
    stream = open(stream_) if READING_FROM_FILE else stream_ 
    q = KLL(space,mode=mode,  c=c)
    for item in stream:
        if CONVERSION_NEEDED:
            q.update(int(item))
        else:
            q.update(item)
    maxError = 0
    if ALL_UNIQUE: 
        for i,j in q.ranks():
            maxError = max(maxError, abs(i - j))
    else:
        algoRanks = np.array([j for i,j in q.ranks()])
        items =   np.array([i for i,j in q.ranks()])
        realRanks = findRanks(stream_, items)
        for i in range(len(algoRanks)):
            maxError = max(maxError, abs(algoRanks[i] - realRanks[i]))
    return [ maxError, np.sqrt(q.cumVar), np.sqrt(q.cumVarS)]

def testKLL(mode, nProcesses, reps, stream, spaceRange, c=2./3.):
    Mean = []
    Std = []
    AveCumStd = []
    AveCumStdS = []
    for space in spaceRange:
        # print ("space:", space) 
        runKLLPartial = partial(runKLL, mode=mode, space=space, stream_=stream, c=c)
        res = np.array(runManyReps(nProcesses, reps, runKLLPartial))
        Mean.append(np.mean(res[:,0]))
        Std.append(np.std(res[:,0]))
        AveCumStd.append(np.mean(res[:,1])) 
        AveCumStdS.append(np.mean(res[:,2]))
    print(mode, Mean) 
    print("-",mode, Std) 
    print("--",mode, AveCumStd) 
    print("---",mode, AveCumStdS) 
    return [mode, Mean, Std, AveCumStd, AveCumStdS]

def runLWYCLazy(_=0, space=128, stream_=[]):
    stream = open(stream_) if READING_FROM_FILE else stream_ 
    #print (mode,space,stream, _)
    q = LWYCLazy(s = space)
    for  item in stream:
        if CONVERSION_NEEDED:
            q.update(int(item))
        else:
            q.update(item)
    maxError = 0
    if ALL_UNIQUE: 
        for i,j in q.ranks():
            maxError = max(maxError, abs(i - j))
    else:
        algoRanks = np.array([j for i,j in q.ranks()])
        items =   np.array([i for i,j in q.ranks()])
        realRanks = findRanks(stream_, items)
        for i in range(len(algoRanks)):
            maxError = max(maxError, abs(algoRanks[i] - realRanks[i]))
    return maxError

def testLWYCLazy(nProcesses, reps, stream, spaceRange):
    Mean = []
    Std = []
    for space in spaceRange:
        # print ("space:", space) 
        runLWYCLazyPartial = partial(runLWYCLazy, space=space, stream_=stream)
        res = np.array(runManyReps(nProcesses, reps, runLWYCLazyPartial))
        Mean.append(np.mean(res[:]))
        Std.append(np.std(res[:]))
    print("lwyc_lazy", Mean) 
    print("-","lwyc_lazy", Std) 
    return ["lwyc_lazy", Mean, Std]


#def runLWYC(_=0, space=128, stream=[]):
#    if READING_FROM_FILE:
#        stream = open(stream) 
#    #print (mode,space,stream, _)
#    q = LWYC(s = space)
#    for  item in stream:
#        if CONVERSION_NEEDED:
#            q.update(int(item))
#        else:
#            q.update(item)
#    maxError = 0
#    for i,j in q.ranks():
#        maxError = max(maxError, abs(i - j))
#    return maxError
#
#def testLWYC(nProcesses, reps, stream, spaceRange):
#    Mean = []
#    Std = []
#    for space in spaceRange:
#        # print ("space:", space) 
#        runLWYCPartial = partial(runLWYC, space=space, stream=stream)
#        res = np.array(runManyReps(nProcesses, reps, runLWYCPartial))
#        Mean.append(np.mean(res[:]))
#        Std.append(np.std(res[:]))
#    print("lwyc", Mean) 
#    print("-","lwyc", Std) 
#    return ["lwyc", Mean, Std]

def genModeQueue(fname,greedylazy, onetoss, varopt, onepair):
    fd = open(fname,'w')
    for i4 in onepair:
        for i3 in varopt:
            for i2 in onetoss:
                for i1 in greedylazy:
                    fd.write("".join(map(str,[2*i1,i1,i2,i3,i4])) + "\n")



def findRanks(stream, items):
   # print (items) 
    items = np.sort(items)
    ranks = np.zeros(len(items))
    if READING_FROM_FILE:
        stream = open(stream) 
    for item in stream:
        if CONVERSION_NEEDED:
            item = int(item)
        idx = bisect.bisect_right(items, item)
        ranks[idx:] += 1
    return ranks


#def runOneStreamSize(_ = 0, algo="KLL", mode= (0,0,0,0,0), space=512,streamSizeP=5 ): 
#    stream = np.array(range(10**streamSizeP))
#    np.random.seed(seed=1234567)
#    np.random.shuffle(stream)
#    q = LWYC(s = space) if algo=="LWYC" else KLL(space,mode=mode, n=len(stream))
#    for  item in stream:
#        q.update(item)
#    maxError = 0
#    for i,j in q.ranks():
#        maxError = max(maxError, abs(i - j))
#    return [maxError, 0 ,0] if algo=="LWYC" else [maxError, q.cumVar, q.cumVarS]
#
#
#
#def testStreamSizes(algo, reps, nProcesses, space, mode):
#    outputMean = []
#    outputStd = []
#    outputAveCumStd = []
#    outputAveCumStdS = []
#    for streamSizeP in range(5,10):
#    #    stream = np.array(range(10**streamSizeP))
#     #   np.random.seed(seed=1234567)
#     #   np.random.shuffle(stream)
#        pool = Pool(processes=nProcesses)
#        runOneStreamSizePartial = partial(runOneStreamSize,algo=algo, mode=mode, space=space, streamSizeP=streamSizeP)
#        results = pool.map(runOneStreamSizePartial, range(reps))
#    #    results = [] 
#     #   for i in range(reps):
#      #      results.append(runOneStreamSizePartial(i))
#        pool.close()
#        pool.join()
#
#        errors = []
#        cumstds = []
#        cumstdss = []
#        for res in results:
#            errors.append(res[0])
#            cumstds.append(res[1])
#            cumstdss.append(res[2])
#        print (streamSizeP,np.mean(errors), np.std(errors),np.mean(cumstds),np.mean(cumstdss))
#






if __name__ == "__main__":
    
    exp = "caida"
    #stream = Data.load("./datasets/mg5.npy")     
    #stream = open("r5.csv")   
    #stream = "/srv/ssd/quant/caida/flow.csv"   
    #stream = "./datasets/ips.csv"  
    reps = 20 
    nProcesses = 21
    
    if (exp=="testing"):
        CONVERSION_NEEDED = 0
        ALL_UNIQUE= 0
        READING_FROM_FILE = 0
        spaceRange =[4096] 
        testKLL([2,1,1,0,4], nProcesses, reps, stream, spaceRange)
        testLWYCLazy(nProcesses, reps, stream, spaceRange)
    
    if (exp=="gendata"): 
        stream = np.array(range(10**5))
        np.random.seed(seed=1234567)
        np.random.shuffle(stream)
        f = open("r5.csv","w")
        for item in stream:
            f.write(str(item) + "\n")
        f.close()
    
    if (exp=="mode"):
        #testing different modes 
        CONVERSION_NEEDED = 0
        ALL_UNIQUE= 1
        READING_FROM_FILE = 0
        spaceRange =[64,128,256,512,1024,2048,4096] 
        modes = []
        genModeQueue('modes.q', [0,1],[0,1],[0,1],[0,4])
        for mode in open('modes.q'):
            modes.append([int(i) for i in list(mode.rstrip())])
        
        if int(sys.argv[1]) < 16:
            mode = modes[int(sys.argv[1])] 
            testKLL(mode, nProcesses, reps, stream, spaceRange)
        elif int(sys.argv[1]) == 16: 
            testLWYCLazy(nProcesses, reps, stream, spaceRange)
        elif int(sys.argv[1]) == 17: 
            testLWYC(nProcesses, reps, stream, spaceRange)
    
    if (exp=="modesotherdata"):
        #testing different datasets (LWYC, KLL and two combo tricks)
        CONVERSION_NEEDED = 0
        ALL_UNIQUE= 0
        READING_FROM_FILE = 0
        spaceRange =[64,128,256,512,1024,2048,4096] 
        modes = [[0,0,0,0,0],[2,1,1,0,4], [2,1,0,1,4]]
        reps = 20
        nProcesses = 21
        streams = ["./datasets/g6.npy","./datasets/mg6.npy","./datasets/zi6.npy","./datasets/zo6.npy","./datasets/s6.npy","./datasets/sq6.npy"] 
        stream = Data.load(streams[int(sys.argv[1])]) 
        print(streams[int(sys.argv[1])]) 
        testKLL(modes[0], nProcesses, reps, stream, spaceRange)
        testKLL(modes[1], nProcesses, reps, stream, spaceRange)
        testKLL(modes[2], nProcesses, reps, stream, spaceRange)
        testLWYCLazy(nProcesses, reps, stream, spaceRange)

    if (exp=="errorvsstreamlength"):
        #testing different datasets (LWYC, KLL and two combo tricks)
        CONVERSION_NEEDED = 0
        ALL_UNIQUE= 1
        READING_FROM_FILE = 0
        spaceRange =[64,128,256,512,1024,2048,4096] 
        modes = [[0,0,0,0,0],[2,1,1,0,4], [2,1,0,1,4]]
        reps = 20
        nProcesses = 21
        streams = ["./datasets/r5.npy","./datasets/r6.npy","./datasets/r7.npy","./datasets/r8.npy","./datasets/r9.npy"] 
        if int(sys.argv[1]) < 3: 
            for stream in streams:
                print(stream) 
                stream = Data.load(stream) 
                testKLL(modes[int(sys.argv[1])], nProcesses, reps, stream, spaceRange)
                #testKLL(modes[1], nProcesses, reps, stream, spaceRange)
               # testKLL(modes[2], nProcesses, reps, stream, spaceRange)
        else: 
            for stream in streams:
                print(stream) 
                stream = Data.load(stream) 
                testLWYCLazy(nProcesses, reps, stream, spaceRange)



    if (exp=="caida"):
        #testing different datasets (LWYC, KLL and two combo tricks)
        CONVERSION_NEEDED = 0
        ALL_UNIQUE= 0
        READING_FROM_FILE = 1
        spaceRange =[20,128,256,512,1024,2048,4096] 
        modes = [[0,0,0,0,0],[2,1,1,0,4], [2,1,0,1,4]]
        reps = 20
        nProcesses = 21
        stream = "./datasets/flow.csv"  
        #stream = "./datasets/ips.csv"  
        if int(sys.argv[1]) < 3: 
            testKLL(modes[int(sys.argv[1])], nProcesses, reps, stream, spaceRange)
        else: 
            testLWYCLazy(nProcesses, reps, stream, spaceRange)



    if (exp=="paramC"): 
        #testing parameter c
        mode1 = [0,0,0,0,0] 
        mode2 = [2,1,1,0,4]
        spaceRange =[4096] 
        for c in np.arange(0.25, 1,0.01):
            print(c)
            testKLL(mode1, nProcesses, reps, stream, spaceRange, c=c)
        
        for c in np.arange(0.25, 1,0.01):
            print(c)
            testKLL(mode2, nProcesses, reps, stream, spaceRange, c=c)
             
    if (exp=="errVSstreamlength"): 
        #testing parameter c
        spaceRange =[4096]
        for mode in modes:
            for n in np.arange(5, 6):
                print("stream length 10^", n)
                stream = Data.load("./datasets/r" + str(n) + ".npy") 
                testKLL(mode, nProcesses, reps, stream, spaceRange, c=c)
    
        
    
