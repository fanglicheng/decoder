#!/usr/bin/env python3

import sys
from os import system

if __name__ == '__main__':
    config = sys.argv[1]
    f = open(config)
    for line in f:
        if '_Nodes' in line:
            nodes = eval(line.strip().split('=')[1])
    f.close()

    for n in nodes:
        system('ssh %s killall %s' % (n, 'decoder.py'))  
