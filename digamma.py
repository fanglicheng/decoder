#!/usr/bin/env python3
from math import log, exp

def digamma(x):
  result = 0
  while x < 7:
      result -= 1/float(x)
      x += 1
  x -= 1.0/2.0
  xx = 1.0/x
  xx2 = xx*xx
  xx4 = xx2*xx2
  result += (log(x) + 
             (1./24.)*xx2 -
             (7.0/960.0)*xx4 +
             (31.0/8064.0)*xx4*xx2 -
             (127.0/30720.0)*xx4*xx4)
  return result

if __name__ == '__main__':
    x = 0.1
    while x < 10:
        print("digamma(%s) = %s, exp(digamma(%s)) = %s" %
              (x, digamma(x), x, exp(digamma(x))))
        x += 0.1
