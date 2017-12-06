import numpy as np
import random
import logging
from math import sqrt
import time


class DataStream:
    def __init__(self, n, order=''):
        self.i = -1
        self.n = n
        self.BP = 1000000007
        orders = ['sorted', "reverse", 'zoomin', 'zoomout', 'sqrt', 'random', 'test']
        assert(order in orders)
        self.order = order;
        if self.order == "zoomin" or self.order == "zoomout":
            self.flip = 0
        if self.order == "random":
            random.seed(self.BP)
            self.hash = random.randint(0, self.BP)


    def next(self):
        if order == "sorted":
            self.i += 1
            return self.i
        elif order == "reverse":
            self.i += 1
            return self.n - self.i
        elif order == "zoomin":
            self.flip = (self.flip + 1) % 2
            if self.flip:
                self.i += 1
                return self.i
            else:
                return self.n - self.i
        elif order == "zoomout":
            self.flip = (self.flip + 1) % 2
            if self.flip:
                self.i += 1
                return n/2 + self.i
            else:
                return n/2 - self.i
        elif order == "random":
            self.i += 1
            return self.hash*self.i%self.BP%self.n

        elif order == "test":
            self.i += 2
            return n/2 - self.i

    def ranks(self, nums):
        self.i = -1
        hist = np.zeros((len(nums),))
        for i in xrange(self.n):
            item = self.next()
            for num_i, num in enumerate(nums):
                if item < num:
                    hist[num_i] += 1
        return hist

    def ranksError(self,nums1, nums2):
        return np.max(np.abs(np.devide(np.array(nums1) - np.array(nums2)), nums1))

if __name__ == '__main__':
    ##testing
    n = 100
    order = "random"
    ds = DataStream(n, order)
    for i in xrange(n):
        print(ds.next())

    print ds.ranks([45])
