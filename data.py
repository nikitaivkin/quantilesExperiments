import numpy as np
import random
import logging
from math import sqrt
import argparse, sys

class Data:
    @staticmethod
    def load(path):
        data = np.load(path)
        return data

    @staticmethod
    def onTheFly(n, order=''):
        orders = ['sorted', 'random']
        assert (order in orders)
        if order == 'sorted': 
            for item in range(n):
                yield item
        if order == 'random':  
            items = list(range(n))
            random.shuffle(items)
            for item in items:
                yield item

    @staticmethod
    def getQuantiles(data, nums):
        return np.searchsorted(np.sort(data), nums)

    @staticmethod
    def gen2file(path, n, order, binary):
        data = np.zeros(n)
        for item_i, item in enumerate(Data.onTheFly(n, order)):
            if item_i >= n:
                break;
            data[item_i] = item
        if binary == "binary": 
            np.save(path, data)
        else:
            np.savetxt(path, data, fmt="%u")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=int, default=6, 
                        help='''length of the stream (set x to get 10^x)''')
    parser.add_argument('-o', type=str, default="random", 
                         help='''stream order: random/sorted''')
    parser.add_argument('-f', type=str, default="length_order", 
                         help='''path to the output file''')
    parser.add_argument('-b', type=str, default="text", 
                         help='''output file binary/text''')
    args = parser.parse_args()
    length = args.l ; order = args.o; filename = args.f; binary = args.b;
    if filename == "length_order":
        filename = str(length) + "_" + order    
    Data.gen2file(filename, 10**length, order, binary)
    
   
