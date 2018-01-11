#!/usr/bin/env python3

from sys import stdout
from common import count_lines

class PercentCounter(object):
    """this counter shows progress of a long running task"""
    def __init__(self, total=-1, input='', file=stdout):
        assert total == -1 or input == '', \
                "user should specify either 'total' or 'input'"
        if total != -1:
            self.total = total
        elif input:
            self.total = count_lines(input)
        else:
            assert False, "please specify either 'total' or 'input'"
        self.percent = 0
        self.file = file

    def print_percent(self, i):
        if i/self.total*100 > self.percent + 1:
            self.percent += 1 
            self.file.write('\r%s%%'.ljust(3) % self.percent)
            if i == self.total - 1:
                self.file.write('\r')
            self.file.flush()
