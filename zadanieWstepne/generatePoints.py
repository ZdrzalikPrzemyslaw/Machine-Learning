import numpy as np
from matplotlib import pyplot as plt
from itertools import product


def main():
    n = 100
    var1, var2 = (np.random.rand(n) * 10) - 5, (np.random.rand(n) * 10) - 5
    c, c1, c2 = [], [], []
    if len(var1) != len(var2):
        exit(1)
    else:
        for i in range(len(var1)):
            c.append((var1[i], var2[i]))
    print(c)
    var3 = np.poly1d([2, -2])
    for i in c:
        if var3(i[0]) >= i[1]:
            c1.append(i)
        else:
            c2.append(i)
    c = np.asarray(c1)
    x, y = c.T
    plt.scatter(x, y)
    c = np.asarray(c2)
    x, y = c.T
    plt.scatter(x, y)
    z = np.linspace(-5, 5, num=10)
    fx = []
    for i in z:
        fx.append(var3(i))
    plt.plot(z, fx)
    plt.show()
    with open("punkt.txt", "w") as plikPL:
        for i in c1:
            plikPL.write(str(i[0]) + " " + str(i[1]) + " 0\r\n")
        for i in c2:
            plikPL.write(str(i[0]) + " " + str(i[1]) + " 1\r\n")


if __name__ == "__main__":
    main()
