import argparse

def arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_test = subparsers.add_parser('load-matrix')
    parser_test.add_argument('input')
    
    return parser.parse_args()

def load_matrix(opts):
    with open(opts.input, 'r') as f:
        print('import numpy as np')
        print('MUNSELL_DATA = np.array([')
        idx = 380
        first = True
        for line in f:
            if idx % 5 == 0 and idx <= 700:
                if idx == 380:
                    if first:
                        first = False
                    else:
                        print(',')
                    print('    [ ', end='')
                else:
                    print(', ', end='')
                num = line.strip()
                print(num, end='')
                if idx == 700:
                    print(' ]', end='')
            idx += 1
            if idx > 800:
                idx = 380
        print('])')

def main():
    opts = arguments()
    if opts.command == 'load-matrix':
        load_matrix(opts)

if __name__ == '__main__':
    main()
