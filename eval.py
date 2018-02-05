from klls import *
from lwyc import *
from data import *
import numpy as np
import sys



def testLWYC(reps, stream):
    outputMean = []
    outputStd = []
    # outputAveCumStd = []
    for space in [128,256,512,1024,2048,4096]:
        errors = []
        # cumstds = []
        for i in range(reps):
            q = LWYC(s = space)
            for  item in stream:
                q.update(item)
            maxError = 0
            for i,j in q.ranks():
                maxError = max(maxError, abs(i - j))
            errors.append(maxError)
            # cumstds.append(np.sqrt(q.cumVar))
        # outputAveCumStd.append(np.mean(cumstds))
        outputMean.append(np.mean(errors))
        outputStd.append(np.std(errors))
    mode = "lwyc"
    res = [mode, outputMean, outputStd]#, outputAveCumStd]
    print(res[0], res[1][0], res[1][1], res[1][2], res[1][3],res[1][4],res[1][5])
    print("-", res[0], res[2][0], res[2][1], res[2][2], res[2][3],res[2][4],res[2][5])
    # print("--", res[0], res[3][0], res[3][1], res[3][2], res[3][3],res[3][4],res[3][5])
    return [mode, outputMean, outputStd]#, outputAveCumStd]


def runOneMode(mode, reps, stream):
    outputMean = []
    outputStd = []
    outputAveCumStd = []
    outputAveCumStdS = []
    for space in [128,256,512,1024,2048,4096]:
        errors = []
        cumstds = []
        cumstdss = []
        # print ("space:", space) 
        for i in range(reps):
            # print ("rep_i:", i)
            q = KLL(space,mode=mode, n=len(stream))
            for  item in stream:
                q.update(item)
            maxError = 0
            for i,j in q.ranks():
                maxError = max(maxError, abs(i - j))
            errors.append(maxError)
            cumstds.append(np.sqrt(q.cumVar))
            cumstdss.append(np.sqrt(q.cumVarS))
        outputAveCumStd.append(np.mean(cumstds))
        outputAveCumStdS.append(np.mean(cumstdss))
        outputMean.append(np.mean(errors))
        outputStd.append(np.std(errors))
    return [mode, outputMean, outputStd, outputAveCumStd, outputAveCumStdS]


def testModes(modes, reps, stream, nProcesses):
    pool = Pool(processes=nProcesses)
    runOneModePartial = partial(runOneMode, reps=reps, stream=stream)
    results = pool.map(runOneModePartial, modes)
    pool.close()
    pool.join()
    for res in results:
        print(res[0], res[1][0], res[1][1], res[1][2], res[1][3],res[1][4],res[1][5])
        print("-", res[0], res[2][0], res[2][1], res[2][2], res[2][3],res[2][4],res[2][5])
        print("--", res[0], res[3][0], res[3][1], res[3][2], res[3][3],res[3][4],res[3][5])
        print("---", res[0], res[4][0], res[4][1], res[4][2], res[4][3],res[4][4],res[4][5])

    # for mode in modes:
    #     [mode, outputMean, outputStd, outputAveCumStd] = runOneMode(mode, reps, stream)
    #     print(str(mode) + "\t" + str(outputMean[0]) + "\t" + str(outputMean[1])+ "\t" + str(outputMean[2]) + "\t" + str(outputMean[3]))
    #     print(str(mode) + "\t" + str(outputStd[0]) + "\t" + str(outputStd[1])+ "\t" + str(outputStd[2]) + "\t" + str(outputStd[3]))
    #     print(str(mode) + "\t" + str(outputAveCumStd[0]) + "\t" + str(outputAveCumStd[1])+ "\t" + str(outputAveCumStd[2]) + "\t" + str(outputAveCumStd[3]))


def genModeQueue(fname,greedy, lazy, onetoss, varopt, onepair):
    fd = open(fname,'w')
    for i4 in onepair:
        for i3 in varopt:
            for i2 in onetoss:
                for i1 in lazy:
                    for i0 in greedy:
                        fd.write("".join(map(str,[i0,i1,i2,i3,i4])) + "\n")


def runOneC(c, mode, space, reps, stream):
    errors = []
    output = []
    for baseLayerSize in [4,8,16,32,64,128]:
        for i in range(reps):
            q = KLL(s=space, c=c, mode=mode, n=len(stream))
            for  item in stream:
                q.update(item)
            maxError = 0
            for i,j in q.ranks():
                maxError = max(maxError, abs(i - j))
            errors.append(maxError)
        output.append([mode, c, baseLayerSize, np.mean(errors), np.std(errors)])
    return output


def testCs(Cs, mode, reps, space, stream, nProcesses):
    pool = Pool(processes=nProcesses)
    runOneCPartial = partial(runOneC,mode=mode, space=space, reps=reps, stream=stream)
    results = pool.map(runOneCPartial, Cs)
    pool.close()
    pool.join()

    for res in results:
        for ins in res:
            print(ins[0], ins[1], ins[2], ins[3], ins[4])
    # for c in Cs:
    #     [mode, c, errorMean, errorStd] = runOneCPartial(c=c)
    #     print(str(c) + "\t" + str(errorMean) + "\t" + str(errorStd))
    #     # print(myMode)



def runOneStreamSize(_ = 0, algo="KLL", mode= (0,0,0,0,0), space=512,streamSizeP=5 ): 
    stream = np.array(range(10**streamSizeP))
    np.random.seed(seed=1234567)
    np.random.shuffle(stream)
    q = LWYC(s = space) if algo=="LWYC" else KLL(space,mode=mode, n=len(stream))
    for  item in stream:
        q.update(item)
    maxError = 0
    for i,j in q.ranks():
        maxError = max(maxError, abs(i - j))
    return [maxError, 0 ,0] if algo=="LWYC" else [maxError, q.cumVar, q.cumVarS]



def testStreamSizes(algo, reps, nProcesses, space, mode):
    outputMean = []
    outputStd = []
    outputAveCumStd = []
    outputAveCumStdS = []
    for streamSizeP in range(5,10):
    #    stream = np.array(range(10**streamSizeP))
     #   np.random.seed(seed=1234567)
     #   np.random.shuffle(stream)
        pool = Pool(processes=nProcesses)
        runOneStreamSizePartial = partial(runOneStreamSize,algo=algo, mode=mode, space=space, streamSizeP=streamSizeP)
        results = pool.map(runOneStreamSizePartial, range(reps))
    #    results = [] 
     #   for i in range(reps):
      #      results.append(runOneStreamSizePartial(i))
        pool.close()
        pool.join()

        errors = []
        cumstds = []
        cumstdss = []
        for res in results:
            errors.append(res[0])
            cumstds.append(res[1])
            cumstdss.append(res[2])
        print (streamSizeP,np.mean(errors), np.std(errors),np.mean(cumstds),np.mean(cumstdss))






if __name__ == "__main__":
    streamFiles= ["./datasets/r5.npy","./datasets/s5.npy", "./datasets/zi5.npy", "./datasets/zo5.npy", "./datasets/sq5.npy" ]
    stream = Data.load(streamFiles[int(sys.argv[1])])
    # print(stream[:100])
    #stream = np.array(range(10**5))
    #np.random.seed(seed=1234567)
    #np.random.shuffle(stream)
    reps = 20 
    nProcesses = 20
    modes = []
    genModeQueue('modes.q', [0,2],[0,1],[0,1],[0,1],[0,1,3,4])
    for mode in open('modes.q'):
        modes.append([int(i) for i in list(mode.rstrip())])
    #print (modes)
    #testModes(modes[:], reps, stream, nProcesses)
    #testLWYC(reps, stream ) 
    #Cs = list((np.array(range(10,90)))/100.)
    # # print(Cs)
    space = 512
    #mode = (0,0,0,0,0)
    #testCs(Cs, mode, reps, space, stream, nProcesses)
    #mode = (1,1,1,0,4)
    #testCs(Cs, mode, reps, space, stream, nProcesses)
    #mode = (0,0,0,0,0)
    #algo ="KLL"
    #testStreamSizes(algo,reps, nProcesses, space, mode)
    #mode = (1,1,1,0,4)
    #algo ="KLL"
    #testStreamSizes(algo,reps, nProcesses, space, mode)
    #algo ="LWYC"
    #testStreamSizes(algo,reps, nProcesses, space, mode)
