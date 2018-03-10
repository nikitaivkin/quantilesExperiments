from __future__ import print_function
import sys
from random import random
from random import randint
from math import ceil, log, sqrt, factorial
import numpy as np
import time
import cProfile
# import logging
import bisect
from heapq import merge
from multiprocessing import Pool
from functools import partial

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# handler = logging.FileHandler(__name__ + '.log')
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)


class KLL(object):
    def __init__(self, s=128, c=2.0 / 3.0, mode=(0,0,0,0,0), n=None, baseLayerSize=4):
        self.mode = mode
        self.greedyMode, self.lazyMode, self.oneTossMode,\
                         self.varOptMode, self.onePairMode = mode
        self.s = s                       # max number of counters to use at any time

        self.k = int(self.s*(1-c) + 4)   # k is a top layer size, 
                                         # 4 -is a bottom layer size
        self.c = c                       # layer size decreasing rate
        self.compactors = []
        self.H = 0                       # number of layers (height)
        self.maxH = log(self.k/4, 1./c)  # max number of layers (max height)

        self.size = 0                    # current number of counters in use
        self.maxSize = 0                 # max number of counters that 
                                         # can be used in current configuration of KLL
        self.D = 0                       # depth (defines the sampling rate)
        self.sampler = Sampler()         # sampler object
        self.cumVar = 0                  # cumulative variance introduced from compactions
        self.cumVarS = 0                 # cumulative variance introduced from compactions and sampling
        self.grow()                      # create first empty compactor
        # logger.info('CREATED: with s,k,c, maxH, mode = ' + ", ".join(map(str,[self.s, self.k,self.c, self.maxH, self.mode])) )
    def kll2str(self):
        s = ''
        for c in self.compactors[::-1]:
            s += str(c) + "\n"
        return s

    def compactRule(self):
        # logger.info('CHECKSIZE: size=' + str(self.size) +' maxSize= ' + str(self.maxSize) + '  mode=' + str(self.mode))
        if self.greedyMode == 0:
            return (len(self.compactors[0]) >= self.capacity(0) or self.size >= self.s)
        if self.greedyMode == 1:
            return self.size >= self.maxSize
        if self.greedyMode == 2:
            return self.size >= self.s
    
    def grow(self):
        self.compactors.append(Compactor(self.mode))
        if self.H + 1 > self.maxH:
            self.D += 1
            self.compactors.pop(0)
        else:
            self.H += 1
        self.maxSize = min(sum(self.capacity(height) for height in range(self.H)),self.s)

    def capacity(self, hight):
        depth = self.H - hight - 1
        return int(ceil(self.c ** depth * self.k))

    def update(self, item):
        if (self.sampler.sample(item, self.D)):
            bisect.insort_left(self.compactors[0],item)
            self.size += 1
            if self.compactRule():
                self.compress()
            assert self.size < self.s, "overi2"
            self.addCumVarS()
    
    def addCumVarS(self):
        if self.D > 0:
            self.cumVarS += (2**self.D)**2/2
    
    def addCumVar(self, h):
        if not self.onePairMode:
            self.cumVar += float((2 ** (h+ self.D))**2)/ float((2**(self.oneTossMode + self.varOptMode)))
            self.cumVarS += float((2 ** (h+ self.D))**2)/ float((2**(self.oneTossMode + self.varOptMode)))
        elif self.onePairMode == 1:
            self.cumVar += float((2 ** (h+ self.D))**2)/float(len(self.compactors[h]))
            self.cumVarS += float((2 ** (h+ self.D))**2)/float(len(self.compactors[h]))
        elif self.onePairMode == 2:
            # technically not true, but real value is difficult to compute
            self.cumVar += float((2 ** (h+ self.D))**2)/float(len(self.compactors[h]))
            self.cumVarS += float((2 ** (h+ self.D))**2)/float(len(self.compactors[h]))
        elif self.onePairMode == 3 or self.onePairMode == 4:
            # adding cumVar only after entire layer was sweeped
            pair_i = bisect.bisect_left(self.compactors[h], self.compactors[h].prevCompInd)
            if pair_i > len(self.compactors[h]) - 2:
                self.cumVar += (2 ** (h+ self.D))**2/ float(2**(self.oneTossMode))
                self.cumVarS += (2 ** (h+ self.D))**2/ float(2**(self.oneTossMode))

    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H:
                    self.grow()
                    if h + 1 >= self.H:
                        h-=1
                if self.onePairMode:
                    bisect.insort_left(self.compactors[h+1],self.compactors[h].compactPair()[0])
                else:
                    self.compactors[h + 1].extend(self.compactors[h].compactLayer())
                    self.compactors[h + 1].sort()
               # self.addCumVar(h)

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
        #print (itemsAndWeights)
        itemsAndWeights.sort()
        cumWeight = 0
        prev_item = None 
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            if item !=prev_item:
                ranksList.append((item, cumWeight))
            prev_item = item
        return ranksList


class Compactor(list):
    def __init__(self, mode):
        super(Compactor, self).__init__()
        self.oneTossMode, self.varOptMode, self.onePairMode = mode[2:]
        self.randBit = random() < 0.5 # compact even or odd
        self.prevCompRand = 1 # flag = 1 if prev compaction was random
        self.randShift = (random() < 0.5)*self.varOptMode   # compact 0:k-1 or 1:k
        self.prevCompInd = None # index of previously compacted pair, or previously compacted value

    def compactLayer(self):
        # self.sort()
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
            self.randBit = random() < 0.5
        elif self.onePairMode == 1:
            randChoice = randint(0,len(self) - 2)
            pair = [self.pop(randChoice), self.pop(randChoice)]
            # pair = [self.pop(randint(0,len(self) - 2)), self.pop(randint(0,len(self) - 2))]
            self.randBit = random() < 0.5
#        elif self.onePairMode == 2:
#            pair_i = min(self.prevCompInd, len(self) - 2)
#            pair = [self.pop(pair_i), self.pop(pair_i)]
#            if self.prevCompInd + 1 <= len(self) - 2:
#            # if self.prevCompRand + 1 >= len(self) - 2:
#                self.prevCompInd += 1
#            else:
#                self.prevCompInd = 0
#            self.randBit = random() < 0.5
        elif self.onePairMode == 3:
            pair_i = bisect.bisect_left(self, self.prevCompInd) if self.prevCompInd != None else 0
            if pair_i > len(self) - 2:
                pair_i = 0
            pair = [self.pop(pair_i), self.pop(pair_i)]
            self.prevCompInd = pair[1]
            self.randBit = random() < 0.5
        elif self.onePairMode == 4:
            pair_i = bisect.bisect_left(self, self.prevCompInd) if self.prevCompInd != None else 0
            if pair_i > len(self) - 2:
                pair_i = 0
                if self.oneTossMode and self.prevCompRand:
                    self.randBit =  not self.randBit
                else:
                    self.randBit = random() < 0.5
                self.prevCompRand = not self.prevCompRand
            pair = [self.pop(pair_i), self.pop(pair_i)]
            self.prevCompInd = pair[1]

        _out = [pair[self.randBit]]
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

