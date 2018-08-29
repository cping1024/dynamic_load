#coding=utf-8

import sys

def main():
    if len(sys.argv) != 2:
        print("Invalid Args!")
        return

    bash = sys.argv[0]
    args = sys.argv[1]
    print("py shell: {0}".format(bash))
    print("input args: {0}".format(args))
    return
    
if __name__ == "__main__":
    main()
