#!/u/lfang/python3/bin/python3.1

import sys

from rule import Rule
from percent_counter import PercentCounter

def usage():
    print("""%s TEST_FILE RULE_FILE
The script writes filtered rules to stdout.
TEST_FILE: sentence-per-line file of test sentences.
RULE_FILE: file with one rule per line.
""" % sys.argv[0])
    sys.exit(1)

if __name__ == '__main__':
    for i, line in enumerate(sys.stdin):
        fields = line.split('|||')
        tmp = fields[1]
        fields[1] = fields[2]
        fields[2] = tmp
        sys.stdout.write(' ||| '.join(fields))
