#!/usr/bin/env python3

from time import sleep
from os import system
import os

import logger

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]

def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

class Monitor(object):
    def __init__(self, node=None):
        self.node = node
        pid = os.getpid()
        self.tmp_stat = '/tmp/tmp_stat_%s' % pid
        self.tmp_meminfo = '/tmp/tmp_meminfo_%s' % pid

    def probe(self, cpu_reserve, mem_reserve):
        """return number of decoder instances that can run on this node."""
        result = 0
        if self.node:
            status = system('ssh %s :' % self.node)
            if status != 0:
                logger.writeln('%s down' % self.node)
            else:
                cpu = self.cpu_usage()
                mem = self.mem_free()
                if logger.level >= 1:
                    logger.writeln('cpu usage: %.1f%%' % cpu)
                    logger.writeln('mem free: %s kB' % mem)
                result = int(min((100-cpu)/cpu_reserve, mem/mem_reserve))
        if logger.level >= 1:
            logger.writeln('%s decoder instances will start on %s' %
                           (result, self.node))
            logger.writeln() 
        system('rm -f %s' % self.tmp_stat)
        system('rm -f %s' % self.tmp_meminfo)
        return result

    def get_stat_file(self, node=None):
        if node:
            system('ssh %s cat /proc/stat > %s' % (node, self.tmp_stat))
        else:
            system('cp /proc/stat %s' % self.tmp_stat)

    def get_meminfo_file(self, node=None):
        if node:
            system('ssh %s cat /proc/meminfo > %s' % (node, self.tmp_meminfo))
        else:
            system('cp /proc/meminfo %s' % self.tmp_meminfo)

    def cpu_usage(self, interval=1):
        times1 = self.parse_proc_stat()
        sleep(interval)
        times2 = self.parse_proc_stat()
        idle = times2[3] - times1[3]
        total = sum(t2 - t1 for t2, t1 in zip(times2, times1))
        return float(total - idle)/total*100

    def parse_proc_stat(self):
        self.get_stat_file(self.node)
        f = open(self.tmp_stat)
        l = f.readline()
        f.close()
        l = l.split()
        user_time = int(l[1])
        nice_time = int(l[2])
        sys_time = int(l[3])
        idle_time = int(l[4])
        return user_time, nice_time, sys_time, idle_time

    def mem_free(self):
        self.get_meminfo_file(self.node)
        f = open(self.tmp_meminfo)
        data = {}
        for line in f:
            line = line.split()
            name = line[0][:-1]
            value = int(line[1])
            data[name] = value
        # print('MemTotal: %s' % data['MemTotal'])
        # print('MemFree: %s' % data['MemFree'])
        # print('Buffers: %s' % data['Buffers'])
        # print('Cached: %s' % data['Cached'])
        # print('SwapCached: %s' % data['SwapCached'])
        # print('Active: %s' % data['Active'])
        # print('Inactive: %s' % data['Inactive'])
        # print('Buffers+Cached+SwapCached: %s' % (data['Buffers'] + data['Cached'] + data['SwapCached']))
        # print('Buffers+Cached+SwapCached+MemFree: %s' % (data['Buffers'] + data['Cached'] + data['SwapCached'] + data['MemFree']))
        # print('Active+Inactive: %s' % (data['Active'] + data['Inactive']))
        # print('Inactive+MemFree: %s' % (data['Inactive'] + data['MemFree']))
        free = data['MemFree'] + data['Buffers'] + data['Cached']
        return free

if __name__ == '__main__':
    m = Monitor()
    m.probe(25, 100000)
    for i in range(30):
        node = 'f%s' % str(i).rjust(2, '0')
        print('probe %s' % node)
        m = Monitor(node)
        print(m.probe(25, 1000000))
