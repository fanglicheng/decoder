#!/usr/bin/env python3
from heapq import heapify, heappop, heappush

class Cube(object):
    """
    See the example below:

    >>> c = Cube()
    >>> c.add_cube([[1,3,7],[1,2,5,7]], sum)
    >>> c.add_cube([[2,3],[5]], sum)
    >>> c.add_cube([[1,3],[]], sum)
    >>> for x in c.iter_top(3):
    ...    print(x)
    ... 
    2
    3
    4
    >>> for x in c.iter_top(20):
    ...    print(x)
    ... 
    2
    3
    4
    5
    6
    7
    8
    8
    8
    8
    9
    10
    12
    14
    """

    def __init__(self):
        self.cubes = []
        self.ops = []  # combining operators for each cube
        self.result = []  # ranked items popped from queue
        # remember enqueued items so we don't enqueue the same item twice
        self.enqueued_items = set()
        self.queue = []  # queue of generated items
        
        # remember the last pop from the queue, so we are a bit more lazy:
        # neighbors are not enqueued until more items are requested
        self.ci = None  # latest cube index
        self.cv = None  # latest vector that points to an item in a cube

        # when all_cubes_added, do heapify
        self.heapified = False

    def add_cube(self, cube, op):
        """'cube' is a list of ordered lists to be combined.
        'op' takes one item from each cube and outputs a new item."""
        ci = len(self.cubes)  # cube index of this cube
        self.cubes.append(cube)
        self.ops.append(op)

        # initialize priority queue with cube tip
        try:
            # interestingly, this also works for the boundary case when
            # len(cube)==0, that is, a zero degree edge on a hypergraph
            # op works on an empty list in this case, and cv == []
            # and no further items will be popped from this cube because
            # cv can't be expanded
            item = op([l[0] for l in cube])
            cv = (0,)*len(cube)  # index within cube
            self.queue.append((item, ci, cv))
            self.enqueued_items.add((ci, cv))
        except IndexError:
            pass

    def __getitem__(self, n):
        """return n'th best item inside the cubes. raise IndexError if n is 
        out of range"""
        if not self.heapified:
            heapify(self.queue)
            self.heapified = True

        while n > len(self.result) - 1:
            # enqueue neighbors
            if self.ci is not None:
                cube = self.cubes[self.ci]
                op = self.ops[self.ci]
                for d in range(len(cube)):
                    cv = list(self.cv)
                    cv[d] += 1
                    cv = tuple(cv)
                    try:
                        item = op([l[cv[d]] for d, l in enumerate(cube)])
                        if (self.ci, cv) not in self.enqueued_items:
                            heappush(self.queue, (item, self.ci, cv))
                            self.enqueued_items.add((self.ci, cv))
                    except IndexError:
                        pass
            # pop top item
            if self.queue:
                top_item, self.ci, self.cv = heappop(self.queue)
                self.result.append(top_item)
            else:
                raise IndexError
        return self.result[n]

    def iter_top(self, n):
        """Pop no more than n new items from cubes"""
        i = 0
        while i < n:
            try:
                yield self[i]
            except IndexError:
                return
            i += 1
    
    def __str__(self):
        result = '<class Cube>\n'
        indent = ' '*4
        for i, cube in enumerate(self.cubes):
            result += 'cube %s\n' % i
            for j, l in enumerate(cube):
                result += '%slist %s\n' % (indent, j)
                for item in l:
                    result += '%s%s\n' % (indent*2, item)
        return result

if __name__ == '__main__':
    import doctest
    doctest.testmod()
