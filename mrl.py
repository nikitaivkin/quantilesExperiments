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


class MRL:
    def __init__(self, s = None,c = None, mode= None,n= None):
        self.k, self.b = 10,10 # to do the function calculating k and b
        self.findKB(s, n)
        self.alBuckets = []   #active layer buckets
        self.tlBuckets = []    # top layer buckets
        self.alBuckets.append(BucketM(self.k, 1))
        self.decisionBit = 0
        self.cumVar = 0             # dummy variable just to keep interfaces the same

    def update(self,item):
        if len(self.alBuckets[-1]) < self.k:
            self.alBuckets[-1].append(item)
        else:
            if len(self.alBuckets) + len(self.tlBuckets) == self.b:
                tmp = BucketM(self.k, None, self.alBuckets, decisionBit = self.decisionBit)
                self.decisionBit = (self.decisionBit+1)%2
                self.alBuckets = []
                self.tlBuckets.append(tmp)
                if len(self.tlBuckets) == self.b - 1:
                    self.alBuckets[:] = self.tlBuckets[:]
                    self.tlBuckets = []
            self.alBuckets.append(BucketM(self.k, 1))
            self.alBuckets[-1].append(item)


    def ranks(self):
        allItems = []
        for bucket in self.alBuckets:
            allItems.extend(zip(bucket[:], np.ones(self.k)*bucket.w))
        for bucket in self.tlBuckets:
            allItems.extend(zip(bucket[:], np.ones(self.k) * bucket.w))
        allItems.sort(key=lambda x: x[0])
        allItems = np.array(allItems)
        allItems[:,1] = np.cumsum(allItems[:,1])
        return allItems


    def findKB(self, s, n):
        minEps = 1
        minB = None
        for b in range(5,30):
            k = s/b
            cond = True
            maxH = 2
            h = 2
            while cond:
                h*=1.1
                cond = self.NchK(int(b+h-2),int(h-1)) < n/k
            h = int(h)
            eps = float((h-2)*self.NchK(b+h-3,h-1) - self.NchK(b+h-3,h-3) - self.NchK(b+h-3,h-2)) /n
            if eps > 0 and eps < minEps:
                minEps = eps
                minB = b
        self.b = minB
        self.k = s/minB
        # print(minEps)

    def NchK(self,n,k):
        return factorial(n)/ (factorial(k) * factorial(n-k))



class BucketM(list):
    def __init__(self, k, bWeight, buckets=None, decisionBit=None):
        super(BucketM, self).__init__()
        self.w = bWeight
        if buckets is not None:
            self.w = sum([b.w for b in buckets])

            allItems = []
            for bucket in buckets:
                allItems.extend(zip(bucket[:], np.ones(k) * bucket.w))
            allItems.sort(key=lambda x: x[0])
            allItems = np.array(allItems)
            allItems[:, 1] = np.cumsum(allItems[:, 1])
            if self.w % 2 == 1:
                idxs = np.arange(k)*self.w +(self.w+1)/2
            else:
                idxs = np.arange(k) * self.w + (self.w + 2*decisionBit) / 2
            self[:] = allItems[np.searchsorted(allItems[:,1], idxs),0]

def test():
    # # q = MRL(96, 10**7)
    # modes = [(0,0,0,0,0,0), (0,0,0,0,1,0),(0,0,0,1,0,0),(0,0,0,1,1,0),(0,0,1,0,0,0),(0,0,1,0,1,0)]
    # modes.extend([(0,0,1,1,0,0), (0,0,1,1,1,0),(0,1,0,0,1,0),(0,1,0,1,0,0),(0,1,0,1,1,0),(0,1,1,0,0,0)])
    # modes.extend([(0,1,1,0,1,0), (0,1,1,1,0,0),(0,1,1,1,1,0)])
    # modes.extend([(1,0,0,0,0,0), (1,0,0,0,1,0),(1,0,0,1,0,0),(1,0,0,1,1,0),(1,0,1,0,0,0),(1,0,1,0,1,0)])
    # modes.extend([(1,0,1,1,0,0), (1,0,1,1,1,0),(1,1,0,0,1,0),(1,1,0,1,0,0),(1,1,0,1,1,0),(1,1,1,0,0,0)])
    # modes.extend([(1,1,1,0,1,0), (1,1,1,1,0,0),(1,1,1,1,1,0)])
    # modes.extend([(0,0,0,0,0,1), (0,0,1,0,0,1),(0,1,1,0,0,1),(1,0,0,0,0,1), (1,0,1,0,0,1),(1,1,1,0,0,1)]) 
    # modes.extend([(0,0,0,0,0,2), (0,0,1,0,0,2),(0,1,1,0,0,2),(1,0,0,0,0,2), (1,0,1,0,0,2),(1,1,1,0,0,2)])
    
    modes = [(0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,3)]
    #modes = [(0,1,1,1,1,0),(1,1,1,1,1,0),(2,1,1,1,1,0)]
    a = np.array(range(100000))
    np.random.shuffle(a)
    for myMode in modes:
        print(myMode)
        for i in xrange(20):
            q = KLL(96,mode=myMode, n =10**5)
            # # q = CormodeRandom(96, Noneu)
            # # # q = Quant5(96, 2./3)
            for item_i,item in enumerate(a):
                q.update(item)
                # if item_i%10000 == 0:
                #     print(item_i)
            maxError = 0
            for i,j in q.ranks():
                maxError = max(maxError, abs(i - j))
            print(maxError)

if __name__ == "__main__":
    test()
