import numpy
import time
from Layer import Layer

# wspolczynnik uczenia
eta = 0.6
# momentum
alfa = 0.2


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        return "hidden_layer (wiersze - neurony) :\n" + str(self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer)

    # liczba neuronów w warstwie ukrytej, liczba neuronów na wyjściu, liczba inputów
    # Tworzymy sobie dwa matrixy
    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs):
        self.hidden_layer = (2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)).T - 1)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs, number_of_neurons_hidden_layer)).T
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)).T - 1
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer, number_of_neurons_output)).T

    def sigmoid_fun(self, inputcik):
            return 1 / (1 + numpy.exp(-inputcik))

    # z wolfram alpha
    def sigmoid_fun_deriative(self, inputcik):
        return numpy.exp(-inputcik) /  ((numpy.exp(-inputcik) + 1) ** 2)

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    def calculate_outputs(self, inputs):
        hidden_layer_output = []
        #print(self.hidden_layer.T)
        for i in self.hidden_layer:
            hidden_layer_output.append(numpy.dot(inputs, i))
        hidden_layer_output = self.sigmoid_fun(numpy.asarray(hidden_layer_output))
        output_layer_output = []
        for i in self.output_layer:
            output_layer_output.append(numpy.dot(i, hidden_layer_output))
        output_layer_output = self.sigmoid_fun(numpy.asarray(output_layer_output))
        return hidden_layer_output, output_layer_output

    #trening, tyle razy ile podamy epochów
    def train(self, inputs, expected_outputs, epoch_count):
        for it in range(epoch_count):

            # Shuffle once each iteration
            joined_arrays = numpy.concatenate((inputs, expected_outputs), axis=1)
            numpy.random.shuffle(joined_arrays)
            joined_arrays_left, joined_arrays_right = numpy.hsplit(joined_arrays, 2)
            numpy.testing.assert_array_equal(joined_arrays_left, joined_arrays_right)
            for k, j in zip(joined_arrays_left, joined_arrays_right):

                hidden_layer_output, output_layer_output = self.calculate_outputs(k)

                output_error = j - output_layer_output
                output_delta = output_error * self.sigmoid_fun_deriative(output_layer_output)

                hidden_layer_error = []
                for i in self.output_layer.T:
                    hidden_layer_error.append(i.dot(output_delta))
                hidden_layer_error = numpy.asarray(hidden_layer_error)
                hidden_layer_delta = hidden_layer_error * self.sigmoid_fun_deriative(hidden_layer_output)

                output_layer_adjustment = []
                for i in output_delta:
                    output_layer_adjustment.append(hidden_layer_output * i)
                output_layer_adjustment = numpy.asarray(output_layer_adjustment)

                hidden_layer_adjustment = []
                for i in hidden_layer_delta:
                    hidden_layer_adjustment.append(k * i)
                hidden_layer_adjustment = numpy.asarray(hidden_layer_adjustment)

                # print(self.hidden_layer)
                # print(self.delta_weights_hidden_layer)

                self.hidden_layer += eta * hidden_layer_adjustment + alfa * self.delta_weights_hidden_layer
                self.output_layer += eta * output_layer_adjustment + alfa * self.delta_weights_output_layer

                self.delta_weights_hidden_layer = -hidden_layer_adjustment
                self.delta_weights_output_layer = -output_layer_adjustment

                if it % 100 == 0:
                    print("iteration - ", it)
                    print(k)
                    print(j)
                    print(output_layer_output)

                    # print("hidden_layer_output\n", hidden_layer_output, "\n", "output_layer_output\n", output_layer_output
                    #       , "\noutput_error\n", output_error, "\noutput_delta\n", output_delta
                    #       , "\nhidden_layer_error\n", hidden_layer_error, "\nhidden_layer_delta\n", hidden_layer_delta
                    #       , "\nhidden_layer_adjustment\n", hidden_layer_adjustment, "\noutput_layer_adjustment\n", output_layer_adjustment)


# funkcja zwraca 2d array intów w postaci arraya z paczki numpy.
def read_2d_int_array_from_file(file_name):
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
    # liczba neuronów w warstwie ukrytej, liczba wyjść, liczba inputów
    siec = NeuralNetwork(3, 4, 4)
    # print(siec)


    # print(siec)
    siec.train(read_2d_int_array_from_file("dane.txt"), read_2d_int_array_from_file("dane.txt").T, 100000)
    # print("Wynik:")
    # inpuciki = read_2d_int_array_from_file("dane.txt")[0]
    # print(inpuciki)
    inpuciki = numpy.asarray([0, 1, 0, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([1, 0, 0, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([0, 0, 1, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([0, 0, 0, 1])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])


if __name__ == "__main__":
    main()

