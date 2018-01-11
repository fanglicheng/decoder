#!/usr/bin/env python3

import sys
from hypergraph import Node, Deserializer
import hypergraph
from decode import Deduction

def decode(hg_file, weights):
    deserializer = Deserializer(Node, Deduction)
    hg = deserializer.deserialize(hg_file)
    for edge in hg.edges():
        edge.cost = sum(f*w for f,w in zip(edge.fcosts, weights))
    hg.set_semiring(hypergraph.SHORTEST_PATH)
    hg.set_functions(lambda x: x.cost, None, None)
    kbest = hg.root.best_paths()
    return ' '.join(kbest[0].translation[1:-1])
    # TODO: no rule in hypergraph file

if __name__ == '__main__':
    weights = [float(x) for x in sys.argv[1].split(',')]
    hg_files = sys.argv[2:]
    for hg_file in hg_files:
        print(decode(hg_file, weights))
