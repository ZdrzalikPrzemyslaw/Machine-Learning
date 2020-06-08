import numpy
import time
# import bigfloat

import matplotlib.pyplot as plt

# wspolczynnik uczenia
eta = 0.1
# momentum
alfa = 0.1


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        if self.is_bias:
            return "hidden_layer (wiersze - neurony) :\n" + str(
                self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(
                self.output_layer) + "\nbiashiddenlayer\n" + str(
                self.bias_hidden_layer) + "\nbiasoutputlayer\n" + str(self.bias_output_layer)
        return "hidden_layer (wiersze - neurony) :\n" + str(
            self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer)

    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs, is_bias):
        # czy uruchomilismy bias
        self.is_bias = is_bias
        # warstwy ukryta i wyjściowa oraz odpowiadające im struktury zapisujące zmianę wagi w poprzedniej iteracji, używane do momentum
        self.hidden_layer = (2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)).T - 1)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs, number_of_neurons_hidden_layer)).T
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)).T - 1
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer, number_of_neurons_output)).T
        # jesli wybralismy że bias ma byc to tworzymy dla każdej warstwy wektor wag biasu
        if is_bias:
            self.bias_hidden_layer = (2 * numpy.random.random(number_of_neurons_hidden_layer) - 1)
            self.bias_output_layer = (2 * numpy.random.random(number_of_neurons_output) - 1)
        # jesli nie ma byc biasu to tworzymy takie same warstwy ale zer. Nie ingerują one potem w obliczenia w żaden sposób
        else:
            self.bias_hidden_layer = numpy.zeros(number_of_neurons_hidden_layer)
            self.bias_output_layer = numpy.zeros(number_of_neurons_output)
        # taka sama warstwa delty jak dla layerów
        self.bias_output_layer_delta = numpy.zeros(number_of_neurons_output)
        self.bias_hidden_layer_delta = numpy.zeros(number_of_neurons_hidden_layer)

    # Wzór funkcji
    def sigmoid_fun(self, inputcik):
        return 1 / (1 + numpy.exp(-inputcik))

    # interesujące jest to, że według mojej wiedzy te wzory są równe sobie a dają dość bardzo różne wyniki w niektórych przypadkach
    # z wolfram alpha
    # def sigmoid_fun_deriative(self, inputcik):
    #     return numpy.exp(-inputcik) /  ((numpy.exp(-inputcik) + 1) ** 2)

    def sigmoid_fun_deriative(self, inputcik):
        return inputcik * (1 - inputcik)

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    # Jak wiadomo bias to przesunięcie wyniku o stałą więc jeżeli wybraliśmy że bias istnieje to on jest po prostu dodawany do odpowiedniego wyniku iloczynu skalarnego
    def calculate_outputs(self, inputs):

        hidden_layer_output = self.sigmoid_fun(numpy.dot(inputs, self.hidden_layer.T) + self.bias_hidden_layer)
        output_layer_output = numpy.dot(hidden_layer_output, self.output_layer.T) + self.bias_output_layer

        return hidden_layer_output, output_layer_output

    # trening, tyle razy ile podamy epochów
    # dla każdego epochu shufflujemy nasze macierze i przechodzimy przez nie po każdym wierszu z osobna
    def train(self, inputs, expected_outputs, epoch_count):
        error_list = []
        for it in range(epoch_count):

            # Shuffle once each iteration
            joined_arrays = numpy.vstack((inputs, expected_outputs)).T
            numpy.random.shuffle(joined_arrays)
            joined_arrays_left, joined_arrays_right = numpy.hsplit(joined_arrays, 2)
            mean_squared_error = 0
            ite = 0

            for k, j in zip(joined_arrays_left, joined_arrays_right):

                hidden_layer_output, output_layer_output = self.calculate_outputs(k)

                # błąd dla wyjścia to różnica pomiędzy oczekiwanym wynikiem a otrzymanym
                output_error = output_layer_output - j
                mean_squared_error += output_error.dot(output_error) / 2
                ite += 1

                # output_delta - współczynnik zmiany wagi dla warstwy wyjściowej. Otrzymujemy jeden współczynnik dla każdego neronu.
                # aby potem wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                # Pochodna funkcji liniowej = 1
                output_delta = output_error * 1

                # korzystamy z wcześniej otrzymanego współczynniku błędu aby wyznaczyć błąd dla warstwy ukrytej
                hidden_layer_error = output_delta.T.dot(self.output_layer)
                # jak dla warstwy wyjściowej hidden_layer_delta jest jeden dla każdego neuronu i
                # aby wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                hidden_layer_delta = hidden_layer_error * self.sigmoid_fun_deriative(hidden_layer_output)

                output_layer_adjustment = []
                for i in output_delta:
                    output_layer_adjustment.append(hidden_layer_output * i)
                output_layer_adjustment = numpy.asarray(output_layer_adjustment)

                hidden_layer_adjustment = []
                for i in hidden_layer_delta:
                    hidden_layer_adjustment.append(k * i)
                hidden_layer_adjustment = numpy.asarray(hidden_layer_adjustment)

                # jeżeli wybraliśmy żeby istniał bias to teraz go modyfikujemy
                if self.is_bias:
                    hidden_bias_adjustment = eta * hidden_layer_delta + alfa * self.bias_hidden_layer_delta
                    output_bias_adjustment = eta * output_delta + alfa * self.bias_output_layer_delta
                    self.bias_hidden_layer -= hidden_bias_adjustment
                    self.bias_output_layer -= output_bias_adjustment
                    self.bias_hidden_layer_delta = hidden_bias_adjustment
                    self.bias_output_layer_delta = output_bias_adjustment

                # wyliczamy zmianę korzystając z współczynnika uczenia i momentum
                hidden_layer_adjustment = eta * hidden_layer_adjustment + alfa * self.delta_weights_hidden_layer
                output_layer_adjustment = eta * output_layer_adjustment + alfa * self.delta_weights_output_layer

                # modyfikujemy wagi w warstwach
                self.hidden_layer -= hidden_layer_adjustment
                self.output_layer -= output_layer_adjustment

                # zapisujemy zmianę wag by użyć ją w momentum
                self.delta_weights_hidden_layer = hidden_layer_adjustment
                self.delta_weights_output_layer = output_layer_adjustment

            mean_squared_error = mean_squared_error / ite
            error_list.append(mean_squared_error)
        print("OSTATNI BLAD", error_list[-1])
        # po przejściu przez wszystkie epoki zapisujemy błędy średniokwadratowe do pliku
        with open("mean_squared_error.txt", "w") as file:
            for i in error_list:
                file.write(str(i) + "\n")


# otwieramy plik errorów i go plotujemy
def plot_file():
    with open("mean_squared_error.txt", "r") as file:
        lines = file.read().splitlines()
    values = []

    for i in lines:
        values.append(float(i))
    plt.plot(values, markersize=1)
    plt.xlabel('Iteration')
    plt.ylabel('Error for epoch')
    plt.title("Mean square error change")
    plt.show()


def plot_function(siec, title, neurons, points=None):
    if points is not None:
        values = read_2d_float_array_from_file(points)
        values2 = numpy.zeros_like(values)
        indexes = numpy.argsort(values[:, 0])
        for i in range(len(indexes)):
            values2[i] = values[indexes[i]]
        points = values2
        values = []
        plt.plot(points[:, 0], points[:, 1], label="original function")
        points = points[:, 0]
        for i in points:
            values.append(siec.calculate_outputs(i)[1][0][0])
        plt.plot(points, values, 'o', markersize=1, label="aproximation")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("File: " + title[:-4] + ", neuron count = " + str(neurons))
        plt.legend()
        plt.tight_layout()
        plt.show()


# funkcja zwraca 2d array floatów w postaci arraya z paczki numpy.
def read_2d_float_array_from_file(file_name):
    two_dim_list_of_return_values = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        for j in list(map(float, i.split())):
            one_dim_list.append(j)
        two_dim_list_of_return_values.append(one_dim_list)
    return numpy.asarray(two_dim_list_of_return_values)


def main():
    neurons = 12
    # ilość neuronów, ilość wyjść, ilość wejść, czy_bias
    siec = NeuralNetwork(number_of_neurons_hidden_layer=neurons,
                         number_of_neurons_output=1, number_of_inputs=1, is_bias=True)
    train_file = "approximation_train_1.txt"
    iterations = 1000
    # dane wejściowe, dane wyjściowe, ilość epochów
    siec.train(read_2d_float_array_from_file(train_file)[:, 0], read_2d_float_array_from_file(train_file)[:, 1],
               iterations)
    plot_file()
    test_file = "approximation_test.txt"
    plot_function(siec, train_file, neurons, test_file)
    # counter = 0
    # blad = 0
    # for i in read_2d_float_array_from_file("approximation_test.txt"):
    #     blad += ((siec.calculate_outputs(i[0])[1][0][0] - i[1]) ** 2) / 2
    #     counter += 1
    # blad = blad / counter
    # print("BLAD ", blad)


if __name__ == "__main__":
    main()
