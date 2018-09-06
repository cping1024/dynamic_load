#! /usr/bin/python
# -*- coding=UTF-8 -*-

import subprocess, sys

if __name__ == '__main__':
    print("test python3 subprocess")
    process = subprocess.Popen('ls -l', shell=True)
    
#    while True:
#        line = process.stdout.readline
#        if process.poll is not None:
#            break
#
#        if line:
#            sys.stdout.write(str(line))
#            sys.stdout.flush
#    
    process.wait
    print("main process exit!")
