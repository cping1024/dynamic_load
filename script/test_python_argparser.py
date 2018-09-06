#! /usr/bin/python
# -*- coding=UTF-8 -*-

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'test python args parser.')
    parser.add_argument("--input_file", metavar="[input file]", type=str, required=True, dest="input_file", help="the script first args!")
    args = parser.parse_args()
    print(args.input_file)
