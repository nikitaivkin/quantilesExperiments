from random import random, randint
from math import ceil, floor, log, sqrt, factorial
import bisect
import numpy as np
import argparse, sys

class LWYC:
    def __init__(self, s, **kvargs):
        eps = self.space2eps(s)                     # find eps based on space constraints
        self.b = int((log(1./eps,2)) + 1)           # number of buckets
        self.s = int(1./eps*sqrt(log(1./eps,2)))    # bucket size 
        self.buckets = [Bucket() for _ in range(self.b)]  # init buckets array (maintained in non-increasing order of layer)
        self.bucketsLayers = np.array([0]*self.b)   # init buckets layers array
        self.bucket_i = 0                           # index to nonFull bucket where we can add
        self.al = 0                                 # active layer value (defines sampling rate)
        self.sampler = Sampler()                    
        self.count = 0                              # current length of the stream
    
    def update(self,item):
        self.count += 1 
        item = self.sampler.sample(item, self.al)       # sample an item (returns None if not sampled)
        if item is None:
            return 
        
        self.buckets[self.bucket_i].append(item)        # adding to the current bucket
        if len(self.buckets[self.bucket_i]) < self.s:   # if current bucket is not full
            return  
        
        self.bucket_i += 1                              # move to the next bucket 
        if self.bucket_i < self.b:                      # if there is another bucket available
            return                                                       
        
        # finding the lowest layer with two buckets for merging (the most left pair in the array, but with the lowest layer)
        t1 = self.bucketsLayers; t2 = t1.tolist()
        idx = t2.index(t1[np.nonzero(t1[:-1] - t1[1:] == 0)[-1][-1]])
                         
        self.buckets[idx] = Bucket(b1=self.buckets[idx],             # merge found pair (idx,idx+1) into the bucket idx  
                                   b2=self.buckets[idx + 1])        
         
           

        self.buckets[idx + 1:-1] = self.buckets[idx + 2:]            # shift all the buckets so the last one is empty now
        self.bucketsLayers[idx + 1:-1] = self.bucketsLayers[idx + 2:]            
        del self.buckets[-1][:]                                      # last bucket is non-full now
        self.bucket_i -= 1                                           # return index to non-full bucket 
        self.bucketsLayers[idx] += 1                                 # merged one -> layer up
       

        self.al = min(max(0, ceil(log(float(self.count)/(self.s*2**(self.b - 2)),2))), self.bucketsLayers[-2]) # update active layer if needed
        self.bucketsLayers[-1] = self.al 
    
    # returns rank of value in the weighted set of all stored items
    def rank(self, value):
        r = 0
        for (i, c) in enumerate(self.buckets):
            for item in c:
                if item < value:
                    r += 2 ** self.bucketsLayers[i]
        return r

    # returns ranks in the weigjted set of all stored items for all unique stored items 
    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        
        for (i, items) in enumerate(self.buckets):
            itemsAndWeights.extend((item, 2 **  (self.bucketsLayers[i])) for item in items)
        itemsAndWeights.sort()
        cumWeight = 0
        prev_item = None
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            if item!= prev_item:
                ranksList.append((item, cumWeight))
            prev_item = item
        return ranksList
  
    def evalMaxError(self, data):
        estRanks = np.array(self.ranks())
        trueRanks = np.zeros(len(estRanks))
        for i in np.searchsorted(estRanks[:,0], data):
            trueRanks[i-1] += 1 
        trueRanks = np.cumsum(trueRanks)
        estRanks = np.array(estRanks[:,1],dtype=np.int32)
        maxError = max([abs(i-j) for i,j in zip(estRanks, trueRanks)])
        return maxError    
 
    # computes space needed for given eps
    def eps2space(self,eps):
        return sqrt(log(1.0 / eps, 2)) * (log(1.0 / eps, 2) + 1) / eps

    # performs binary search to find \eps according to given space 
    def space2eps(self,s):
        e2s = self.eps2space   
        l = 10**-10; r = 1 - l
        while e2s(l) >= e2s(r)+1:
            m = (l+r)/2
            if s > e2s(m): r= m
            else: l= m
        return l


class Bucket(list):
    #inits an empty bucket if b1=b2=None
    #inits a bucket, by merging buckets b1 and b2   
    def __init__(self, b1=None, b2=None):
        super(Bucket, self).__init__()
        if b1 is not None:
            self.extend(sorted(b1 + b2)[random() < 0.5::2])


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
    parser.add_argument('-t', type=str, choices=["string","int","float"], default='string',
                              help='defines the type of stream items, default="string".' )
    args = parser.parse_args()
    s = args.s if args.s > 0 else 256
    conversions = {'int':int,'string':str,'float':float}
    lwyc = LWYC(s) 
    for line in sys.stdin:
        item = conversions[args.t](line.strip('\n\r'))
        lwyc.update(item)
    
    for (item, rank) in lwyc.ranks():
        print('%f\t%s'%(float(rank)/lwyc.count,str(item)))
