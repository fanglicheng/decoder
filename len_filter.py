#!/usr/bin/env python3

import sys

def output_name(filename, n):
    if '.' in filename:
        name, suffix = filename.rsplit('.', 1)
    else:
        name = filename
    name = '%s_maxlen_%s' % (name, n)
    if '.' in filename:
        return name + '.' + suffix
    else:
        return name

def usage():
    pass
    sys.exit(1)

if __name__ == '__main__':
    max_len = int(sys.argv[1])
    files = [open(filename) for filename in sys.argv[2:]]
    outfiles = [open(output_name(filename, max_len), 'w')
                for filename in sys.argv[2:]]
    while True:
        lines = [f.readline() for f in files]
        if not lines[0]:
            break
        if len(lines[0].split()) <= max_len:
            for line, f in zip(lines, outfiles):
                f.write(line)
    for f in files:
        f.close()
    for f in outfiles:
        f.close()
