import numpy
import time
from Layer import Layer

import matplotlib.pyplot as plt

# wspolczynnik uczenia
eta = 0.6
# momentum
alfa = 0.2


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        if self.is_bias:
            return "hidden_layer (wiersze - neurony) :\n" + str(
                self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer) + "\nbiashiddenlayer\n" + str(
                self.bias_hidden_layer) + "\nbiasoutputlayer\n" + str(self.bias_output_layer)
        return "hidden_layer (wiersze - neurony) :\n" + str(self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer)


    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, number_of_inputs, is_bias):
        # czy uruchomilismy bias
        self.is_bias = is_bias
<<<<<<< HEAD
        self.hidden_layer = (2 * numpy.random.random((number_of_inputs + is_bias, number_of_neurons_hidden_layer)).T - 1)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs + is_bias, number_of_neurons_hidden_layer)).T
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer + is_bias, number_of_neurons_output)).T - 1
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer + is_bias, number_of_neurons_output)).T
        # if is_bias:
        #     self.bias_hidden_layer = (2 * numpy.random.random(number_of_neurons_hidden_layer) - 1)
        #     self.bias_output_layer = (2 * numpy.random.random(number_of_neurons_output) - 1)
        # else:
        #     self.bias_hidden_layer = numpy.zeros(number_of_neurons_hidden_layer)
        #     self.bias_output_layer = numpy.zeros(number_of_neurons_output)
        # self.bias_output_layer_delta = numpy.zeros(number_of_neurons_output)
        # self.bias_hidden_layer_delta = numpy.zeros(number_of_neurons_hidden_layer)
=======
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
>>>>>>> bde85c6e05cc4d8b42fdf3827f76b323d242f212

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



        hidden_layer_output = self.sigmoid_fun(numpy.dot(inputs, self.hidden_layer.T) )

        if self.is_bias:
            hidden_layer_output = numpy.insert(hidden_layer_output, 0, 1)


        output_layer_output = self.sigmoid_fun(numpy.dot(hidden_layer_output, self.output_layer.T))

        return hidden_layer_output, output_layer_output

    # trening, tyle razy ile podamy epochów
    # dla każdego epochu shufflujemy nasze macierze i przechodzimy przez nie po każdym wierszu z osobna
    def train(self, inputs, expected_outputs, epoch_count):
        error_list = []
        if self.is_bias:
            inputs = numpy.insert(inputs, 0, 1, axis=1)
        joined_arrays = list(zip(inputs, expected_outputs))
        for it in range(epoch_count):


            # Shuffle once each iteration
            numpy.random.shuffle(joined_arrays)

            mean_squared_error = 0
            ite = 0

            for k, j in joined_arrays:

                hidden_layer_output, output_layer_output = self.calculate_outputs(k)

<<<<<<< HEAD

                output_error = j - output_layer_output
=======
                # błąd dla wyjścia to różnica pomiędzy oczekiwanym wynikiem a otrzymanym
                output_error = output_layer_output - j
>>>>>>> bde85c6e05cc4d8b42fdf3827f76b323d242f212

                mean_squared_error += output_error.dot(output_error) / 2
                ite += 1

                # output_delta - współczynnik zmiany wagi dla warstwy wyjściowej. Otrzymujemy jeden współczynnik dla każdego neronu.
                # aby potem wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                output_delta = output_error * self.sigmoid_fun_deriative(output_layer_output)
<<<<<<< HEAD

                # print(output_delta)
                # print(self.output_layer.T)
=======
>>>>>>> bde85c6e05cc4d8b42fdf3827f76b323d242f212

                # korzystamy z wcześniej otrzymanego współczynniku błędu aby wyznaczyć błąd dla warstwy ukrytej
                hidden_layer_error = output_delta.T.dot(self.output_layer)
                # jak dla warstwy wyjściowej hidden_layer_delta jest jeden dla każdego neuronu i
                # aby wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                hidden_layer_delta = hidden_layer_error * self.sigmoid_fun_deriative(hidden_layer_output)

                if self.is_bias:
                    hidden_layer_error = hidden_layer_error[1:]
                    hidden_layer_delta = hidden_layer_error * self.sigmoid_fun_deriative(hidden_layer_output[1:])

                output_layer_adjustment = []
                for i in output_delta:
                    output_layer_adjustment.append(hidden_layer_output * i)
                output_layer_adjustment = numpy.asarray(output_layer_adjustment)

                hidden_layer_adjustment = []
                for i in hidden_layer_delta:
                    hidden_layer_adjustment.append(k * i)
                hidden_layer_adjustment = numpy.asarray(hidden_layer_adjustment)

<<<<<<< HEAD
                # if self.is_bias:
                #     hidden_bias_adjustment = eta * hidden_layer_delta + alfa * self.bias_hidden_layer_delta
                #     output_bias_adjustment = eta * output_delta + alfa * self.bias_output_layer_delta
                #     self.bias_hidden_layer += hidden_bias_adjustment
                #     self.bias_output_layer += output_bias_adjustment
                #     self.bias_hidden_layer_delta -= hidden_bias_adjustment
                #     self.bias_output_layer_delta -= output_bias_adjustment

                # print(hidden_layer_adjustment)
                # print(self.delta_weights_hidden_layer)
=======
                # jeżeli wybraliśmy żeby istniał bias to teraz go modyfikujemy
                if self.is_bias:
                    hidden_bias_adjustment = eta * hidden_layer_delta + alfa * self.bias_hidden_layer_delta
                    output_bias_adjustment = eta * output_delta + alfa * self.bias_output_layer_delta
                    self.bias_hidden_layer -= hidden_bias_adjustment
                    self.bias_output_layer -= output_bias_adjustment
                    self.bias_hidden_layer_delta = hidden_bias_adjustment
                    self.bias_output_layer_delta = output_bias_adjustment
>>>>>>> bde85c6e05cc4d8b42fdf3827f76b323d242f212

                #wyliczamy zmianę korzystając z współczynnika uczenia i momentum
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
    plt.xlabel('Epoka')
    plt.ylabel('Wartość błędu')
    plt.plot(values, 'o', markersize=1)
    plt.show()

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
<<<<<<< HEAD
    numpy.random.seed(0)
    # liczba neuronów w warstwie ukrytej, liczba wyjść, liczba inputów
    siec = NeuralNetwork(1, 4, 4, True)
    # print(siec)


    print(siec)
    siec.train(read_2d_int_array_from_file("dane.txt"), read_2d_int_array_from_file("dane.txt").T, 3000)

    plot_file()

    # print("Wynik:")
    # inpuciki = read_2d_int_array_from_file("dane.txt")[0]
    # print(inpuciki)
    # inpuciki = numpy.asarray([0, 1, 0, 0])
    # print(inpuciki)
    # print(siec.calculate_outputs(inpuciki)[1])
    # inpuciki = numpy.asarray([1, 0, 0, 0])
    # print(inpuciki)
    # print(siec.calculate_outputs(inpuciki)[1])
    # inpuciki = numpy.asarray([0, 0, 1, 0])
    # print(inpuciki)
    # print(siec.calculate_outputs(inpuciki)[1])
    # inpuciki = numpy.asarray([0, 0, 0, 1])
    # print(inpuciki)
    # print(siec.calculate_outputs(inpuciki)[1])
    # inpuciki = read_2d_int_array_from_file("dane.txt")
    # print(siec.calculate_outputs(inpuciki)[1])
=======
    # liczba neuronów w warstwie ukrytej, liczba wyjść, liczba inputów, czy_bias
    siec = NeuralNetwork(3, 4, 4, True)

    #dane wejściowe, dane wyjściowe, ilość epochów
    siec.train(read_2d_int_array_from_file("dane.txt"), read_2d_int_array_from_file("dane.txt").T, 5000)

    plot_file()
    print(siec)
    print("Wynik:")
    inpuciki = numpy.asarray([1, 0, 0, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([0, 1, 0, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([0, 0, 1, 0])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
    inpuciki = numpy.asarray([0, 0, 0, 1])
    print(inpuciki)
    print(siec.calculate_outputs(inpuciki)[1])
>>>>>>> bde85c6e05cc4d8b42fdf3827f76b323d242f212


if __name__ == "__main__":
    main()

