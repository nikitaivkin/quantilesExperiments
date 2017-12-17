from __future__ import print_function
import sys
from random import random
from random import randint
from math import ceil, log, sqrt, factorial
import numpy as np
import time
import cProfile
import logging
import bisect
from heapq import merge
from multiprocessing import Pool
from functools import partial

class KLL(object):
    def __init__(self, s=128, c=2.0 / 3.0, mode=(0,0,0,0,0), n=None):
        self.mode = mode
        self.greedyMode, self.lazyMode, self.oneTossMode,\
                         self.varOptMode, self.onePairMode = mode
        
        self.s = s
        self.n = n   # n is needed only for solution without sampling 

        self.k = int(self.s*(1-c) + 4)   # top layer size, 4 -is a bottom layer size
        self.c = c                       # layer size decreasing rate
        self.compactors = []        
        self.H = 0                       # number of layers (height)
        self.maxH = log(self.k/4, 1./c)     # max number of layers (max height)

        self.size = 0               # current number of counters in use
        self.maxSize = 0            # max number of counters that 
                                    # can be used in current configuration 
        self.D = 0                  # depth 
        self.sampler = Sampler()    # sampler object
        self.cumVar = 0             # cumulative variance introduced
        self.grow()                 # create first empty compactor

    def grow(self):
        self.compactors.append(Compactor(self.mode))
        if self.H + 1 > self.maxH:
            self.D += 1
            self.compactors.pop(0)
        else:
            self.H += 1
        self.maxSize = sum(self.capacity(height) for height in range(self.H))

    def capacity(self, hight):
        depth = self.H - hight - 1
        return int(ceil(self.c ** depth * self.k))

    def update(self, item):
        if (self.sampler.sample(item, self.D)):
            bisect.insort_left(self.compactors[0],item)
            self.size += 1
            if(not self.greedyMode and (len(self.compactors[0]) >= self.capacity(0) or (self.size >= self.s))) or \
              (self.greedyMode == 1 and (self.size >= self.maxSize)) or \
              (self.greedyMode == 2 and (self.size >= self.s)):
                self.compress()
                assert self.size < self.s, "over2"
                

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H:
                    self.grow()
                    if h + 1 >= self.H:
                        h-=1
                # assert h >= 0, "under1"
                if self.onePairMode:
                    bisect.insort_left(self.compactors[h+1],self.compactors[h].compactPair()[0])
                else:
                    self.compactors[h + 1].extend(self.compactors[h].compactLayer())
                    # self.compactors[h + 1] = sort(kind='mergesort')
                # if self.varOptMode:
                #     self.cumVar += (2**(h + self.D))**2 / 2
                if self.lazyMode:
                    break
        self.size = sum(len(c) for c in self.compactors)
                


    def rank(self, value):
        r = 0
        for (h, c) in enumerate(self.compactors):
            for item in c:
                if item <= value:
                    r += 2 ** h
        return r


    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
            itemsAndWeights.extend((item, 2 ** (h+ self.D)) for item in items)
 #       print (itemsAndWeights)
        itemsAndWeights.sort()
        cumWeight = 0
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            ranksList.append((item, cumWeight))
        return ranksList


class Compactor(list):
    def __init__(self, mode):
        super(Compactor, self).__init__()
        self.oneTossMode, self.varOptMode, self.onePairMode = mode[2:]
        self.randBit = random() < 0.5 # compact even or odd
        self.prevCompRand = 1 # flag = 1 if prev compaction was random
        self.randShift = (random() < 0.5)*self.varOptMode   # compact 0:k-1 or 1:k
        self.prevCompInd = -1 # index of previously compacted pair

    def compactLayer(self):
        self.sort()
        _in, _out = [], []  # _in stays in compactor; _out is a result of compaction
        _in.extend([self.pop(0)] if (self.randShift and len(self) > 2) else []) # varOptMode requires shift   
        _in.extend([self.pop()] if len(self) % 2 else []) # checking if array to compact is even sized
        _out.extend(self[self.randBit::2])
        self[:] = _in
        if self.oneTossMode and self.prevCompRand:
            self.randBit =  not self.randBit
        else:
            self.randBit = random() < 0.5
        self.prevCompRand = not self.prevCompRand

        if self.varOptMode:
            self.randShift = random() < 0.5
        return _out

    def compactPair(self):
        # self.sort()
        if len(self) == 2:
            pair = [self.pop(), self.pop()]
        elif self.onePairMode == 1:
            randChoice = randint(0,len(self) - 2)
            pair = [self.pop(randChoice), self.pop(randChoice)]
            # pair = [self.pop(randint(0,len(self) - 2)), self.pop(randint(0,len(self) - 2))]
        elif self.onePairMode == 2:
            pair_i = min(self.prevCompInd, len(self) - 2)
            pair = [self.pop(pair_i), self.pop(pair_i)]
            if self.prevCompInd + 1 <= len(self) - 2:
            # if self.prevCompRand + 1 >= len(self) - 2:
                self.prevCompInd += 1
            else:
                self.prevCompInd = 0
        elif self.onePairMode == 3:
            pair_i = bisect.bisect_left(self, self.prevCompInd)
            if pair_i > len(self) - 2:
                pair_i = 0
            pair = [self.pop(pair_i), self.pop(pair_i)]
            self.prevCompInd = pair[1]

        _out = [pair[self.randBit]]
        self.randBit = random() < 0.5
        return _out

class Sampler():
    def __init__(self):
        self.s1 = self.s2 = -1

    def sample(self, item, l):
        if l == 0: return item
        if (self.s2 == -1):
            self.s1 = randint(0, 2 ** l - 1)
            self.s2 = 2 ** l - 1
        self.s1 -= 1
        self.s2 -= 1
        return item if (self.s1 == -1) else None



def runOneMode(mode, space, reps, stream):
    errors = []
    for i in range(reps):
        q = KLL(space,mode=mode, n=len(stream))
        for  item in stream:
            q.update(item)
        maxError = 0
        for i,j in q.ranks():
            maxError = max(maxError, abs(i - j))
        errors.append(maxError)
    return [mode, np.mean(errors), np.std(errors)]

def test(modes, reps, space, stream, nProcesses):
    pool = Pool(processes=nProcesses)
    runOneModePartial = partial(runOneMode, space=space, reps=reps, stream=stream)
    results = pool.map(runOneModePartial, modes)
    pool.close()
    pool.join()

    for res in results:
        print(res[0], res[1], res[2])
    # for mode in modes:
    #     [mode, errorMean, errorStd] = runOneMode(mode, space, reps, stream)
    #     print(str(mode) + "\t" + str(errorMean) + "\t" + str(errorStd))
        # print(myMode)
        # errors = []
        # for i in xrange(20):
        #     q = KLL(96,mode=myMode, n =10**5)
        #     for item_i,item in enumerate(a):
        #         q.update(item)
        #     maxError = 0
        #     for i,j in q.ranks():
        #         maxError = max(maxError, abs(i - j))
        #     errors.append(maxError)

def genModeQueue(fname,greedy, lazy, onetoss, varopt, onepair):
    fd = open(fname,'w')
    for i4 in range(onepair*3 + 1):
        for i3 in range(varopt + 1):
            for i2 in range(onetoss + 1):
                for i1 in range(lazy + 1):
                    for i0 in range(greedy*2 + 1):
                        fd.write("".join(map(str,[i0,i1,i2,i3,i4])) + "\n")






if __name__ == "__main__":
    modes = [(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,3)]
    stream = np.array(range(100000))
    np.random.shuffle(stream)
    reps = 2
    space = 96
    nProcesses = 2
    modes = []
    genModeQueue('modes.q', 1,1,1,1,1)
    for mode in open('modes.q'):
        modes.append([int(i) for i in list(mode.rstrip())])
    # print(modes)
    test(modes, reps, space, stream, nProcesses)
    