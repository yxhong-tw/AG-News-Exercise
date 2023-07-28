import numpy as np


def main():
    a = np.array([1, 1, 1])
    b = np.array([1, 0, 1])

    print((a*b).sum())

if __name__ == '__main__':
    main()