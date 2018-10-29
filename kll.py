from random import random, randint
from math import floor, ceil, log, sqrt, factorial
import bisect
import argparse, sys
import numpy as np

class KLL(object):
    def __init__(self, s= 128, c= 2.0 / 3.0, mode=(0,0,0,0)):
        self.mode = mode
        self.lazy, self.antiCorr,\
            self.innerShift, self.sweep = mode  
                                         # parsing mode options
        self.maxS = s                    # max number of items stored (space limit) 
        self.S = 0                       # current number of items stored 
        self.c = c                       # layer size decreasing rate
        self.blCap = 4                   # Bottom Layer Capacity (the smallest one)  
        self.k = self.blCap      
        
        self.setLayersCaps()             # choses k based on self.maxS, self.blCap and self.c
                                         # sets self.maxH and capacities for all layers self.layersCaps
        self.H = 0                       # current number of layers (height)
        self.D = 0                       # depth (defines the sampling rate)
        self.count = 0                   # current number of updates processed  
        
        self.sampler = Sampler()         
        self.compactors = []            
        self.grow()                      # initialization -- create first empty compactor

    def grow(self):
        self.compactors.append(Compactor(self.mode)) 
        if self.H + 1 > self.maxH:      # if new compactor is too high -> drop the bottom layer
            self.D += 1                  
            self.compactors.pop(0)
        else:
            self.H += 1
    
    # returns max capacity of the compactor at height h
    def capacity(self, h):
        return self.layersCaps[-self.H + h] 
#        return floor(self.k * (self.c ** (self.H - h - 1)))
  
    def update(self, item):
        self.count += 1
        if (self.sampler.sample(item, self.D)) is None: #current item is not sampled
            return 
        bisect.insort_left(self.compactors[0], item)    #insert sampled item into the bottom compactor
        self.S += 1
        if (not self.lazy and len(self.compactors[0]) >= self.capacity(0)) or (self.S >= self.maxS): 
            # compact if needed: (vanilla and current compactor is full) or (lazy and out of memory)
            self.compress()  
    
    def compress(self):
        for h in range(len(self.compactors)):
            if len(self.compactors[h]) >= self.capacity(h):
                if h + 1 >= self.H:  # need to compact top layer
                    self.grow()      # adding new layer above
                    if h + 1 >= self.H: 
                        h-=1         # this happen when H = maxH -> after growing, new layer has height h-1
                if self.sweep:       
                    bisect.insort_left(self.compactors[h+1],self.compactors[h].compactPair())
                else:                 
                    self.compactors[h + 1][:] = sorted(self.compactors[h + 1][:] + self.compactors[h].compactLayer())
                if self.lazy:
                    break            # if lazy -> don't make cascade compactions
        self.S = sum(len(c) for c in self.compactors)

    # returns rank of value v in the weighted set of all stored items
    def rank(self, v):
        r = 0
        for (h, c) in enumerate(self.compactors):
            for item in c:
                if item <= v:
                    r += 2 ** (h+self.D)
        return r

    # returns ranks in the weighted set of all stored items for all unique stored items 
    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.compactors):
            itemsAndWeights.extend((item, 2 ** (h+ self.D)) for item in items)
        itemsAndWeights.sort()
        cumWeight = 0
        prev_item = None 
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            if item !=prev_item:
                ranksList.append((item, cumWeight))
            prev_item = item
        return ranksList

    # finds the best choice of k for given space limit, base layer capacity and parameter C
    def setLayersCaps(self):
        minSpaceLoss = self.maxS 
        self.maxH = 0  
        self.layersCaps = [] 
        for k in range(self.blCap+1,self.maxS): 
            layers = [k] 
            while layers[-1] >= self.blCap and sum(layers) < self.maxS:
                layers.append(ceil(layers[-1]*self.c)) 
            layers = layers[:-1]
            s = sum(layers) 
            if self.maxS - s < minSpaceLoss and len(layers) >= self.maxH:
                self.layersCaps = layers
                minSpaceLoss = self.maxS - s  
                self.maxH = len(layers) 
        self.layersCaps[0] += minSpaceLoss
        self.layersCaps = self.layersCaps[::-1]
   

class Compactor(list):
    def __init__(self, mode):
        super(Compactor, self).__init__()
        self.antiCorr, self.innerShift, self.sweep = mode[1:]
        self.randBit = random() < 0.5                                   # compact even or odd positions
        self.prevCompRand = 1                                           # flag = 1 if prev compaction was random
        self.randShift = (random() < 0.5)*self.innerShift               # compact [0:k-1] or [1:k]
        self.theta = None                                               # previously compacted value (threshold) 
                                                                        # for sweep compactor

    def compactLayer(self):
        # _in stays in compactor; _out is a result of compaction
        _in = [self.pop(0)] if (self.randShift and len(self) > 2) else [] # innerShift requires shift   
        _in.extend([self.pop()] if len(self) % 2 else [])                 # checking if array to compact is even sized
        _out = self[self.randBit::2]                                      # compacting 
        self[:] = _in
        
        self.randBit =  not self.randBit if (self.antiCorr and self.prevCompRand) else random() < 0.5
        self.prevCompRand = not self.prevCompRand
        self.randShift = (random() < 0.5)*self.innerShift
        return _out

    def compactPair(self):
        # compacting the pair of two smallest items  > theta (sweep compacting) 
        idx = bisect.bisect_left(self, self.theta) if self.theta != None else 0
        if idx > len(self) - 2: # new sweep starts 
            idx = self.randShift
            self.randShift = (random() < 0.5)*self.innerShift
            self.randBit =  not self.randBit if (self.antiCorr and self.prevCompRand) else random() < 0.5
            self.prevCompRand = not self.prevCompRand
        pair = [self.pop(idx), self.pop(idx)]
        self.theta = pair[1]
        return pair[self.randBit]


              
class Sampler(): 
    def __init__(self):
        self.s1 = self.s2 = -1
    # out of 2^l consecutive calls
    # return item only once by a random choice which one
    # otherwise return None 
    def sample(self, item, l):
        if l == 0: return item
        if (self.s2 == -1):
            self.s1 = randint(0, 2 ** l - 1)
            self.s2 = 2 ** l - 1
        self.s1 -= 1
        self.s2 -= 1
        return item if (self.s1 == -1) else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=256, 
                        help=''' controls the space usage of the data structure''')
    parser.add_argument('-c', type=float, default=2./3., 
                        help=''' controls the space usage of the data structure''')
    parser.add_argument('-t', type=str, choices=["string","int","float"], default='string',
                        help='defines the type of stream items, default="string".' )
    parser.add_argument('-m', type=str , default='0000',
                        help='mode of the algorithm, lazy (0/1), antiCorr(0/1),' +\
                        'innerShift(0/1), sweeping(0/1), default="0000".' )
    args = parser.parse_args()
    s = args.s ; c = args.c; m =  [int(i) for i in args.m[:]] 
    conversions = {'int':int,'string':str,'float':float}
    kll = KLL(s=s, mode=m, c=c) 
    for line in sys.stdin:
        item = conversions[args.t](line.strip('\n\r'))
        kll.update(item)

    for (item, rank) in kll.ranks():
        print('%f\t%s'%(float(rank)/kll.count,str(item)))


