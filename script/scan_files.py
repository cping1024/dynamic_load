#! /usr/bin/python
# -*- coding=UTF-8 -*-

import os, sys

def scan_files(path, filename):
    f = open(filename, 'w')
    for root, subdir, files in os.walk(path):
        for next_dir in subdir:
             scan_files(next_dir, filename)

        for item in files:
            if item[-3:] == 'jpg' or item[-4:] == 'jpeg' or \
               item[-3:] == 'png' or item[-3:] == 'bmp':
                 f.write(os.path.join(root, item))
                 f.write('\n')
    f.close()

if __name__ == '__main__':
    
    scan_files(path = sys.argv[1], filename = sys.argv[2])
