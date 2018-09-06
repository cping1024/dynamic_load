#! /usr/bin/python
# -*- coding=UTF-8 -*-

import os, sys

def scan_files(path, all_files):
    subdir_list = []
    for root, subdir, files in os.walk(path):
        for next_dir in subdir:
            subdir_list.append(next_dir) 
            #scan_files(next_dir, filename)

        for item in files:
            if item[-3:] == 'jpg' or item[-4:] == 'jpeg' or \
               item[-3:] == 'png' or item[-3:] == 'bmp':
                 all_files.append(os.path.join(root, item))
    
    for directory in subdir_list:
        scan_files(directory, all_files)

def scan_dirs(path, filename):

    all_files = []
    scan_files(path, all_files)
    f = open(filename, 'w')
    for line in all_files:
        f.write(line)
        f.write('\n')

if __name__ == '__main__':
    
    scan_dirs(path = sys.argv[1], filename = sys.argv[2])
