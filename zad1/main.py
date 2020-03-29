import numpy
import time


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        return str(self.hidden_layer) + "\n" + str(self.output_layer)

    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs):
        self.hidden_layer = 2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)) - 1
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)) - 1

    def funkcja_sigmoidalna(self, inputcik):
            return 1 / (1 + numpy.exp(-inputcik))

    def pochodna_funkcja_sigmoidalna(self, inputcik):
        return inputcik * (1 - inputcik)

    def calculate_outputs(self, inputs):
        hidden_layer_output = self.funkcja_sigmoidalna(numpy.dot(inputs, self.hidden_layer))
        output_layer_output = self.funkcja_sigmoidalna(numpy.dot(hidden_layer_output, self.output_layer))
        return hidden_layer_output, output_layer_output

    def train(self, inputs, expected_outputs, epoch_count):
        for it in range(epoch_count):
            hidden_layer_output, output_layer_output = self.calculate_outputs(inputs)

            output_error = expected_outputs - output_layer_output
            output_delta = output_error * self.pochodna_funkcja_sigmoidalna(output_layer_output)


            hidden_layer_error = output_delta.dot(self.output_layer.T)
            hidden_layer_delta = hidden_layer_error * self.pochodna_funkcja_sigmoidalna(hidden_layer_output)

            hidden_layer_adjustment = inputs.T.dot(hidden_layer_delta)
            output_layer_adjustment = hidden_layer_output.T.dot(output_delta)

            self.hidden_layer += hidden_layer_adjustment
            self.output_layer += output_layer_adjustment


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
    # liczba neuronów w warstwie ukrytej, liczba neuronów na wyjściu, liczba inputów
    siec = NeuralNetwork(3, 4, 4)
    # siec.print()
    siec.train(wczytajPunktyZPliku("dane.txt"), wczytajPunktyZPliku("dane.txt").T, 60000)
    print(siec)
    print(siec.calculate_outputs(wczytajPunktyZPliku("dane.txt")))


if __name__ == "__main__":
    main()

