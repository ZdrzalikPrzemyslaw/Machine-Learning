import numpy
from matplotlib import pyplot as plt
import time


class NeuralNetwork:
    def __init__(self):
        self.weight_matrix = 2 * numpy.random.random((4, 1)) - 1

    def funkcja_sigmoidalna(self, inputcik):
            return 1/(1+numpy.exp(-inputcik))

    def pochodna_funckja_sigmoidalna(self, inputcik):
        return numpy.exp(-inputcik) / ((numpy.exp(-inputcik) + 1) ** 2)

    def funkcja_skokowa(self, inputcik):
        wynik = numpy.dot(inputcik, self.weight_matrix)  # iloczyn skalarny tak o świrki
        if wynik >= 0:
            return 1
        return 0

    def train(self, inputs, expected_outputs, amount_of_tries):
        for i in range(0, amount_of_tries):
            wspolczynnik_uczenia = numpy.random.random(1)
            for j in range(0, len(inputs)):
                wynikpl = self.funkcja_skokowa(inputs[j])
                # boze jakie to jest brzydkie ale działa xDDDD
                # jestem pewien że tutaj by styknął iloczyn skalarny ale już huj
                if wynikpl == expected_outputs[j]:
                    pass
                elif wynikpl == 1 and expected_outputs[j] == 0:
                    self.weight_matrix -= wspolczynnik_uczenia * inputs[j]
                elif wynikpl == 0 and expected_outputs[j] == 1:
                    self.weight_matrix += wspolczynnik_uczenia * inputs[j]
                pass
            pass
        pass


def wczytajPunktyZPliku(file_name):
    two_dim_list_of_return_values = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        for j in  list(map(int, i.split())):
            one_dim_list.append(j)
        two_dim_list_of_return_values.append(one_dim_list)
    return numpy.asarray(two_dim_list_of_return_values)
    pass


def main():
    siec = NeuralNetwork()
    print(wczytajPunktyZPliku("dane.txt"))
    print(siec.funkcja_sigmoidalna(wczytajPunktyZPliku("dane.txt")))
    print(siec.pochodna_funckja_sigmoidalna(wczytajPunktyZPliku("dane.txt")))


if __name__ == "__main__":
    main()

