import numpy as np
import argparse, sys

class Data:
    @staticmethod
    def load(path, binary=1):
        return np.load(path) if binary else np.array(open(path).read().splitlines())

    @staticmethod
    def syn2csv(streamType="random", streamLen=4, **kwargs): 
        assert(streamType in ['sorted', 'random', 'trending', 'brownian' ])
        stream = np.arange(10**streamLen) 
        if streamType == 'random':  
            np.random.shuffle(stream)
        elif streamType == 'trending':
            stream = stream + np.random.randint(-10**streamLen *kwargs["trendingP"],10**streamLen*kwargs["trendingP"], 10**streamLen)
        elif streamType == 'brownian':
            stream = np.random.randint(-10**streamLen *kwargs["trendingP"],10**streamLen*kwargs["trendingP"], 10**streamLen)
            stream = np.cumsum(stream) 
        np.savetxt(str(streamLen) + streamType + ".csv",stream, fmt="%u")

    @staticmethod
    def csv2npy(path,t="str"): 
        assert(t in ['str', 'int'])
        lines = open(path).read().splitlines()
        if t == 'int': 
            lines = map(int, lines) 
        stream = np.array(lines)  
        np.save(path[:-4], stream)

    @staticmethod
    def printout(path):
        assert(path[-3:] in ['csv', 'npy'])
        binary = 0 if path[-3:] == "csv" else 1
        data = Data.load(path, binary)  
        for item in data:
            print item

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str,
                        help='''action: 'generate' to generate synthetic dataset, 
                        'convert' to save csv as binary and 'read' to print stream from file ''')
    parser.add_argument('-l', type=int, default=4, 
                        help='''length of the stream (set x to get 10^x)''')
    parser.add_argument('-s', type=str, default="random", 
                        help='''stream type: random/sorted/trending/brownian''')
    parser.add_argument('-p', type=float, default=0.01, 
                        help='''stream generating parameter''')
    parser.add_argument('-t', type=str, default="str", 
                        help='''item type: 'str' or 'int' ''')
    parser.add_argument('-f', type=str, 
                        help='''path to file''')
    
    args = parser.parse_args()
    if args.a  == 'generate':
        Data.syn2csv(streamType=args.s, streamLen=args.l, trendingP=args.p)
    elif args.a  == 'convert' and args.f:
        Data.csv2npy(args.f, t=args.t)
    elif args.a  == 'read' and args.f:
        Data.printout(args.f) 
    
  
