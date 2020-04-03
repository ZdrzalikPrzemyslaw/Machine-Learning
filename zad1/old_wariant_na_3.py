import numpy
import time


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        return str(self.hidden_layer) + "\n" + str(self.output_layer)

    # liczba neuronów w warstwie ukrytej, liczba neuronów na wyjściu, liczba inputów
    # Tworzymy sobie dwa matrixy
    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs):
        self.hidden_layer = 2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)) - 1
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)) - 1

    def funkcja_sigmoidalna(self, inputcik):
            return 1 / (1 + numpy.exp(-inputcik))

    # z wolfram alpha
    def pochodna_funkcja_sigmoidalna(self, inputcik):
        return numpy.exp(-inputcik) /  ((numpy.exp(-inputcik) + 1) ** 2)

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    def calculate_outputs(self, inputs):
        hidden_layer_output = self.funkcja_sigmoidalna(numpy.dot(inputs, self.hidden_layer))
        output_layer_output = self.funkcja_sigmoidalna(numpy.dot(hidden_layer_output, self.output_layer)).T
        return hidden_layer_output, output_layer_output

    #trening, tyle razy ile podamy epochów
    def train(self, inputs, expected_outputs, epoch_count):
        for it in range(epoch_count):
            hidden_layer_output, output_layer_output = self.calculate_outputs(inputs)


            # korzystamy z pochodnej aby policzyć output_delta który wykorzystujemy potem do wyliczenia zmiany wag
            output_error = expected_outputs - output_layer_output
            output_delta = output_error * self.pochodna_funkcja_sigmoidalna(output_layer_output)

            # tutaj jakieś czary mary totalne, patrzymy w jakiś magiczny sposób jak bardzo na błąd w warstwie zewnętrznej
            # miały wpływ wagi w warstwie ukrytej
            hidden_layer_error = output_delta.T.dot(self.output_layer.T)
            hidden_layer_delta = hidden_layer_error * self.pochodna_funkcja_sigmoidalna(hidden_layer_output)

            # wyliczamy zmianę wag
            hidden_layer_adjustment = inputs.T.dot(hidden_layer_delta)
            output_layer_adjustment = hidden_layer_output.T.dot(output_delta.T)

            print("hidden_layer_output\n", hidden_layer_output, "\n",  "output_layer_output\n", output_layer_output
                  , "\noutput_error\n", output_error, "\noutput_delta\n", output_delta
                  , "\nhidden_layer_error\n", hidden_layer_error, "\nhidden_layer_delta\n", hidden_layer_delta
                  , "\nhidden_layer_adjustment\n", hidden_layer_adjustment, "\noutput_layer_adjustment\n", output_layer_adjustment)


            self.hidden_layer += hidden_layer_adjustment
            self.output_layer += output_layer_adjustment

# funkcja zwraca 2d array intów w postaci arraya z paczki numpy.
def wczytajPunktyZPliku(file_name):
    two_dim_list_of_return_values = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        for j in  list(map(int, i.split())):
            one_dim_list.append(j)
        two_dim_list_of_return_values.append(one_dim_list)
    return numpy.asarray(two_dim_list_of_return_values).T


def main():
    # liczba neuronów w warstwie ukrytej, liczba neuronów na wyjściu, liczba inputów
    siec = NeuralNetwork(2, 4, 4)
    # print(siec)
    siec.train(wczytajPunktyZPliku("dane.txt"), wczytajPunktyZPliku("dane.txt").T, 1)
    # print(siec)
    print("Wynik:")
    print(siec.calculate_outputs(wczytajPunktyZPliku("dane.txt"))[1])


if __name__ == "__main__":
    main()