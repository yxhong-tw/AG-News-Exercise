import numpy as np
import torch


def main():
    a = np.array([1, 1, 1])
    b = np.array([1, 0, 1])
    c = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    d = torch.tensor([0.6, 0.4])
    e = []

    for value in c:
        e.append(value.argmax())

    print(e)
    print(type(e))
    print(torch.Tensor(e))
    print(type(e))
    input()

    print(c.argmax())
    print(type(c.argmax()))
    print((a*b).sum())


if __name__ == '__main__':
    main()