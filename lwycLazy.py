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


class LWYCLazy:
    def __init__(self, s = None,c = None, mode= None,n= None):
        eps = self.space2eps(s)
        # print(str(eps) +  " " + str(s))
        self.b = int(ceil((log(1./eps,2) + 1)/2.)*2)
#        print("number of buckets: ", self.b) 
        self.s = 1./eps*sqrt(log(1./eps,2))  # not the total space allowed which is s, but bucket size here
        self.buckets = [BucketCLazy() for _ in range(self.b)] # initializing the buckets itself
        self.bucketsLayers = [0 for _ in range (self.b)]      # initializing layer assigned to each buckets
        self.bucket_i = 0                                     # index to nonFull bucket in Active Layer where we can add
        self.al = 0                                           # active layer value
        self.sampler = SamplerLazy()                          # samler is not lazy this is just to avoid name conflicts
        self.cumVar = 0                                       # dummy variable just to keep interfaces the same 


    def update(self,item):
        item = self.sampler.sample(item, self.al)            # sample an item (returns None if current item is not sampled)
        if item is not None:
            self.buckets[int(self.bucket_i)].append(item)                        # adding to the current bucket
            if len(self.buckets[int(self.bucket_i)]) == int(self.s):             # check if after adding and item bucket got full
                self.bucket_i += 1                                               # if so then moving to the next bucket
                if self.bucket_i > len(self.buckets)-1:                          # if there is no next bucket 
                    idx = bisect.bisect_left(self.bucketsLayers,(-1)* self.al)   # all buckets in self.buckets have decreasing layers 
                                                                                 # layers stored in bucketsLayers as negative numbers
                                                                                 # (to make use of bisect_left)
                                                                                 # to maintain the order we find a pair to merge with 
                                                                                 # the smallest |layer| but earliest in the bucket array
#                    print ("idx: ", idx, "layers: ", self.bucketsLayers ) 
                    self.buckets[idx] = BucketCLazy(b1=self.buckets[idx],        # merge  
                                                    b2=self.buckets[idx + 1])    #
                    self.bucket_i -= 1                                           # we merged to buckets now we should have exactly one emtpy bucket
                    self.buckets[idx + 1][:] = self.buckets[self.bucket_i][:]    # swap idx + 1 and bucket_i, so the last one is empty
                    del self.buckets[self.bucket_i][:]                           # making last one empty
                    self.bucketsLayers[idx] -= 1                                 # merged one -> layer up
                    if idx + 1 == self.bucket_i:                                 # if merged pair is pair of last two buckets, then active layer up
                        self.bucketsLayers[idx+1] -=1
                        self.al += 1

    def rank(self, value):
        r = 0
        for (h, c) in enumerate(self.buckets):
            for item in c:
                if item <= value:
                    r += 2 ** ((-1)*self.bucketsLayers[h])
        return r


    def ranks(self):
        ranksList = []
        itemsAndWeights = []
        for (h, items) in enumerate(self.buckets):
            itemsAndWeights.extend((item, 2 **  ((-1)*self.bucketsLayers[h])) for item in items)
        itemsAndWeights.sort()
        cumWeight = 0
        prev_item = None
        for (item, weight) in itemsAndWeights:
            cumWeight += weight
            if item!= prev_item:
                ranksList.append((item, cumWeight))
            prev_item = item
        return ranksList

  # def ranks(self):
  #      allItems = []
  #      for b in self.buckets:
  #          allItems.extend(b)
  #      allItems.sort()
  #      ranks = np.array(range(len(allItems)))*(2**self.al)
  #      return zip(allItems, ranks)

    def eps2space(self,eps):
        return sqrt(log(1.0 / eps, 2)) * (log(1.0 / eps, 2) + 1) / eps

    def space2eps(self,space):
        left = 0.000001
        right = 0.999999
        while self.eps2space(left) >= self.eps2space(right) + 1:
            midpoint = (left + right) / 2
            if space > self.eps2space(midpoint):
                right = midpoint
            else:
                left = midpoint
        return left


class BucketCLazy(list):   # to accomodate delayed merges it stores its own layer 
    def __init__(self, l=0, b1=None, b2=None):
        super(BucketCLazy, self).__init__()
        self.l = l
        if b1 is not None:
            self.extend(sorted(b1 + b2)[random() < 0.5::2])


class SamplerLazy(): # just to avoid name conflicts - actually it's the same Sampler
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

# def test():
#     # # q = MRL(96, 10**7)
#     # modes = [(0,0,0,0,0,0), (0,0,0,0,1,0),(0,0,0,1,0,0),(0,0,0,1,1,0),(0,0,1,0,0,0),(0,0,1,0,1,0)]
#     # modes.extend([(0,0,1,1,0,0), (0,0,1,1,1,0),(0,1,0,0,1,0),(0,1,0,1,0,0),(0,1,0,1,1,0),(0,1,1,0,0,0)])
#     # modes.extend([(0,1,1,0,1,0), (0,1,1,1,0,0),(0,1,1,1,1,0)])
#     # modes.extend([(1,0,0,0,0,0), (1,0,0,0,1,0),(1,0,0,1,0,0),(1,0,0,1,1,0),(1,0,1,0,0,0),(1,0,1,0,1,0)])
#     # modes.extend([(1,0,1,1,0,0), (1,0,1,1,1,0),(1,1,0,0,1,0),(1,1,0,1,0,0),(1,1,0,1,1,0),(1,1,1,0,0,0)])
#     # modes.extend([(1,1,1,0,1,0), (1,1,1,1,0,0),(1,1,1,1,1,0)])
#     # modes.extend([(0,0,0,0,0,1), (0,0,1,0,0,1),(0,1,1,0,0,1),(1,0,0,0,0,1), (1,0,1,0,0,1),(1,1,1,0,0,1)]) 
#     # modes.extend([(0,0,0,0,0,2), (0,0,1,0,0,2),(0,1,1,0,0,2),(1,0,0,0,0,2), (1,0,1,0,0,2),(1,1,1,0,0,2)])
    
#     modes = [(0,0,0,0,0,0),(0,0,0,0,0,1),(0,0,0,0,0,2),(0,0,0,0,0,3)]
#     #modes = [(0,1,1,1,1,0),(1,1,1,1,1,0),(2,1,1,1,1,0)]
#     a = np.array(range(100000))
#     np.random.shuffle(a)
#     for myMode in modes:
#         print(myMode)
#         for i in xrange(20):
#             q = KLL(96,mode=myMode, n =10**5)
#             # # q = CormodeRandom(96, Noneu)
#             # # # q = Quant5(96, 2./3)
#             for item_i,item in enumerate(a):
#                 q.update(item)
#                 # if item_i%10000 == 0:
#                 #     print(item_i)
#             maxError = 0
#             for i,j in q.ranks():
#                 maxError = max(maxError, abs(i - j))
# #             print(maxError)

# if __name__ == "__main__":
#     test()
