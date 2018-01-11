#!/usr/bin/env python3
from heapq import heapify, heappop, heappush

class LazyListMerger(object):
    """
    See the example below:

    >>> c = LazyListMerger()
    >>> c.add_list([1,3,7])
    >>> c.add_list([1,2,5,7])
    >>> c.add_list([1,3])
    >>> for x in c.iter_top(3):
    ...    print(x)
    ... 
    1
    1
    1
    >>> for x in c.iter_top(20):
    ...    print(x)
    ... 
    1
    1
    1
    2
    3
    3
    5
    7
    7
    """

    def __init__(self):
        self.lists = []
        self.result = []  # ranked items popped from queue
        # remember enqueued items so we don't enqueue the same item twice
        self.queue = []  # queue of enumerated items
        
        # remember the last pop from the queue, so we are a bit more lazy:
        # neighbors are not enqueued until more items are requested
        self.ci = None  # latest list index

        # when all lists are added, do heapify
        self.heapified = False

    def add_list(self, l):
        """'list' is a sorted list"""
        ci = len(self.lists)  # index of this list
        self.lists.append(l)
        # initialize priority queue with list top item
        try:
            item = l[0]
            cv = 0
            self.queue.append((item, ci, cv))
        except IndexError:
            pass

    def __getitem__(self, n):
        """return n'th best item inside the lists. raise IndexError if n is 
        out of range"""
        if not self.heapified:
            heapify(self.queue)
            self.heapified = True

        while n > len(self.result) - 1:
            # enqueue neighbor
            if self.ci is not None:
                l = self.lists[self.ci]
                cv = self.cv + 1
                try:
                    item = l[cv]
                    heappush(self.queue, (item, self.ci, cv))
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
        """Pop no more than n new items from lists"""
        i = 0
        while i < n:
            try:
                yield self[i]
            except IndexError:
                return
            i += 1
    
    def __str__(self):
        result = '<class LazyListMerger>\n'
        indent = ' '*4
        for i, l in enumerate(self.lists):
            result += 'list %s\n' % i
            for item in l:
                result += '%s%s\n' % (indent, item)
        return result

if __name__ == '__main__':
    import doctest
    doctest.testmod()
