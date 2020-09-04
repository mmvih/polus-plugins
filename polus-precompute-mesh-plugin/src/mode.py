import argparse

def main():

    parser = argparse.ArgumentParser(prog='mode', description='Simple code for finding the mode of four values')
    parser.add_argument('--wxyz', dest='wxyz', type=str, nargs='+',
                        help='values for the four pixels', required=True)
    
    args = parser.parse_args()
    values = args.wxyz

    w = values[0]
    x = values[1]
    y = values[2]
    z = values[3]

    output = None
    if w == x:
        if y == z:
            output = max(x,y)
            print(output)
            exit()
        else:
            output = x
            print(output)
            exit()
    else:
        if y == z:
            output = y
            print(output)
            exit()
        else:
            a = min(x,w)
            b = max(x,w)
            x = min(y,z)
            y = max(y,z)
            if b == y:
                output = b
                print(output)
                exit()
            elif b == x:
                output = x
                print(output)
                exit()
            elif x == a:
                output = a
                print(output)
                exit()
            else:
                output = max(b,y)
                print(output)
                exit()

main()
