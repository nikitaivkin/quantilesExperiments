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


class LWYC:
    def __init__(self, s = None,c = None, mode= None,n= None):
        eps = self.space2eps(s)
        # print(str(eps) +  " " + str(s))
        self.b = int(ceil((log(1./eps,2) + 1)/2.)*2)
        self.s = 1./eps*sqrt(log(1./eps,2))  # not the total space allowed which is s, but bucket size here
        self.alBuckets = [BucketC() for _ in range(self.b)]
        self.alBucket_i = 0             #  index to nonFull bucket in Active Layer
        self.al = 0                     #  active layer value
        self.sampler = Sampler()
        self.cumVar = 0                 # dummy variable just to keep interfaces the same 


    def update(self,item):
        item = self.sampler.sample(item, self.al)
        if item is not None:
            self.alBuckets[int(self.alBucket_i)].append(item)
            if len(self.alBuckets[int(self.alBucket_i)]) == int(self.s): # check if after adding and item bucket got full
                self.alBucket_i += 1                                     # if so then moving to the next bucket
                if self.alBucket_i > len(self.alBuckets)-1:              # if there is no next bucket 
                    for i in range(0, int(self.b/2)):                    # merging buckets
                        self.alBuckets[i] = BucketC(self.alBuckets[i],
                                                    self.alBuckets[i+ int(self.b/2)])
                    for b in self.alBuckets[int(self.b/2):]:            # cleaning
                        del b[:]
                    self.alBucket_i = self.b/2
                    self.al += 1

    def ranks(self):
        allItems = []
        for b in self.alBuckets:
            allItems.extend(b)
        allItems.sort()
        ranks = np.array(range(len(allItems)))*(2**self.al)
        return zip(allItems, ranks)

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


class BucketC(list):
    def __init__(self, b1=None, b2=None):
        super(BucketC, self).__init__()
        if b1 is not None:
            self.extend(sorted(b1 + b2)[random() < 0.5::2])


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
