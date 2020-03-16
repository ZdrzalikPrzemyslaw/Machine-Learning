import numpy
import time

class NeuralNetwork():

    def __init__(self):
        # Using seed to make sure it'll
        # generate same weights in every run

        # 2x1 Weight matrix
        self.weight_matrix = 2 * numpy.random.random((3)) - 1

    def funkcja_skokowa(self, inputcik):
        wynik = numpy.dot(inputcik, self.weight_matrix)
        if wynik > 1:
            return 1
        return 0

    def train(self, inputs, expected_outputs, amount_of_tries):
        for i in range(amount_of_tries):
            for j in range(len(inputs)):
                wynikpl = self.funkcja_skokowa(inputs[j])
                if wynikpl == expected_outputs[j]:
                    pass
                elif wynikpl == 1 and expected_outputs[j] == 0:
                    self.weight_matrix -= inputs[j]
                elif wynikpl == 0 and expected_outputs[j] == 1:
                    self.weight_matrix += inputs[j]
                pass
            pass
        pass


def wczytajPunktyZPliku(fileName):
    punkty_pod, punkty_nad = [], []
    with open(fileName, "r") as plik:
        lines = plik.read().splitlines()
    lines = [tup for tup in lines if not tup == ""]
    for i in lines:
        # x, y, z = i.split()\
        liczby = list(map(float, i.split()))
        if liczby[2] == 0:
            punkty_pod.append([liczby[0], liczby[1], 1])
        else:
            punkty_nad.append([liczby[0], liczby[1], 1])
    return punkty_pod, punkty_nad


def main():
    punkty_pod, punkty_nad = wczytajPunktyZPliku("punkt.txt")
    print(punkty_pod)
    print(punkty_nad)
    neural_network = NeuralNetwork()

    print('Random weights at the start of training')
    print(neural_network.weight_matrix)
    super_lista = []
    train_inputs = numpy.array(punkty_pod + punkty_nad)
    for i in punkty_pod:
        super_lista.append(0)
    for i in punkty_nad:
        super_lista.append(1)
    train_outputs = numpy.array([super_lista]).T

    neural_network.train(train_inputs, train_outputs, 1000)

    print('New weights after training')
    print(neural_network.weight_matrix)

    # Test the neural network with a new situation.
    print("Testing network on new examples ->")
    #print(neural_network.forward_propagation(numpy.array([1, 1, 50])))
    print(neural_network.funkcja_skokowa([0, -1, 1]))
    pass


if __name__ == "__main__":
    main()

# import numpy
# import time
#
#
# class NeuralNetwork():
#
#     def __init__(self):
#         # Using seed to make sure it'll
#         # generate same weights in every run
#         numpy.random.seed(1)
#
#         # 2x1 Weight matrix
#         self.weight_matrix = 2 * numpy.random.random((3, 1)) - 1
#
#     def funkcja_skokowa(self, input):
#         wynik = numpy.dot(self.weight_matrix, input)
#         if wynik > 1:
#             return 1
#         return 0
#
#     def train(self, inputs, expected_outputs, amount_of_tries):
#         for i in range(amount_of_tries):
#             for j in range(len(inputs)):
#                 wynikpl = self.funkcja_skokowa(inputs[j])
#                 if wynikpl == expected_outputs[j]:
#                     pass
#                 elif wynikpl == 1 and expected_outputs[j] == 0:
#                     self.weight_matrix -= inputs[j]
#                 elif wynikpl == 0 and expected_outputs[j] == 1:
#                     self.weight_matrix += inputs[j]
#                 pass
#             pass
#         pass
#
#
# def wczytajPunktyZPliku(fileName):
#     punkty_pod, punkty_nad = [], []
#     with open(fileName, "r") as plik:
#         lines = plik.read().splitlines()
#     lines = [tup for tup in lines if not tup == ""]
#     for i in lines:
#         # x, y, z = i.split()\
#         liczby = list(map(float, i.split()))
#         if liczby[2] == 0:
#             punkty_pod.append([1, liczby[0], liczby[1]])
#         else:
#             punkty_nad.append([1, liczby[0], liczby[1]])
#     return punkty_pod, punkty_nad
#
#
# def main():
#     siec = NeuralNetwork()
#     punkty_pod, punkty_nad = wczytajPunktyZPliku("punkt.txt")
#     expected = []
#     for i in punkty_pod:
#         expected.append(0)
#     for i in punkty_nad:
#         expected.append(1)
#     print(numpy.array(punkty_pod + punkty_nad))
#     siec.train(numpy.array(punkty_pod + punkty_nad), numpy.array(expected).T, 10)
#
#
# if __name__ == "__main__":
#     main()