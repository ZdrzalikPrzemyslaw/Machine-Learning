from matplotlib import pyplot as plt


def plot(list2d):
    list1 = []
    list2 = []
    for i in list2d:
        list1.append(i[0])
        list2.append(i[1])
    plt.plot(list1, list2, 'bo')
    plt.show()


def read_file(filename):
    two_dim_list_of_return_values = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        for j in list(map(float, i.split(","))):
            one_dim_list.append(j)
        two_dim_list_of_return_values.append(one_dim_list)
    return sorted(two_dim_list_of_return_values, key=lambda l: l[0])


def main():
    plot(read_file("Danetestowe.txt"))



if __name__ == '__main__':
    main()