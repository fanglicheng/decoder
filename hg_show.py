#!/usr/bin/env python3

import sys

from extractor import Extractor
from alignment import Alignment
import hypergraph
import logger

if __name__ == '__main__':
    ffilename = sys.argv[1] 
    efilename = sys.argv[2] 
    afilename = sys.argv[3] 
    n = int(sys.argv[4])
    ffile = open(ffilename)
    efile = open(efilename)
    afile = open(afilename)
    alignments = Alignment.reader_pharaoh(ffile, efile, afile)
    for i, alignment in enumerate(alignments):
        if i == n:
            extractor = Extractor(maximize_derivation=True)
            hg = extractor.extract_hypergraph(alignment)
            hg.show()
