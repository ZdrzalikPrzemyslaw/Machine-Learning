import numpy
from scipy.spatial import distance

import matplotlib.pyplot as plt

# wspolczynnik uczenia
eta = 0.1
# momentum
alfa = 0


class NeuralNetwork:
    def __repr__(self):
        return "Instance of NeuralNetwork"

    def __str__(self):
        # todo: zaktualizuj to_string()
        if self.is_bias:
            return "hidden_layer (wiersze - neurony) :\n" + str(
                self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(
                self.output_layer) + "\nbiashiddenlayer\n" + str(
                self.bias_hidden_layer) + "\nbiasoutputlayer\n" + str(self.bias_output_layer)
        return "hidden_layer (wiersze - neurony) :\n" + str(
            self.hidden_layer) + "\noutput_layer (wiersze - neurony) :\n" + str(self.output_layer)

    def __init__(self, number_of_neurons_hidden_layer, number_of_neurons_output, is_bias, input_data, expected_outputs):
        # czy uruchomilismy bias, bias aktualnie nie jest zaimplementowany dla warstwy radialnej
        self.is_bias = is_bias

        # dane wejsciowe
        self.input_data = input_data
        self.expected_outputs = expected_outputs

        # Pozycja centrów ma być losowana z wektórów wejściowych
        # Laczymy dane wejsciowe i expected outputs żeby móc je razem przelosować i zachować łączność danych
        input_data_random_order = numpy.vstack((self.input_data, self.expected_outputs)).T
        numpy.random.shuffle(input_data_random_order)

        # wtworzymy wagi dla warstwy wejsciowej, najpierw tworzymy macierz o jakim chcemy rozmiarze

        self.hidden_layer = numpy.zeros((len(input_data_random_order[0, :-1]), number_of_neurons_hidden_layer)).T

        # ustawiamy n neuronom ich centra jako n pierwszych danych wejściowych (po przelosowaniu danych wejsciowych)
        for i in range(numpy.size(self.hidden_layer, 0)):
            self.hidden_layer[i] = input_data_random_order[i, :-1]
        # print(self.hidden_layer)

        # Ustawiamy sigmy początkowo na 1
        self.scale_coefficient = numpy.ones(numpy.size(self.hidden_layer, 0))

        # Szukamy sigm ze wzoru
        self.find_sigma()
        # print(self.scale_coefficient)

        # delty dla momentum, aktualnie nie uczymy wsteczną propagacją warstwy ukrytej więc nie używamy
        self.delta_weights_hidden_layer = numpy.zeros((len(input_data_random_order[0]),
                                                       number_of_neurons_hidden_layer)).T

        # tworzymy warstwę wyjściową z losowymi wagami od -1 do 1, jak w zad 1
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)).T - 1
        # print(self.output_layer)
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

    # szukamy sigm ze wzorów dla każdego neuronu radialnego
    def find_sigma(self):
        for i in range(numpy.size(self.hidden_layer, 0)):
            max_dist = 0
            for j in range(numpy.size(self.hidden_layer, 0)):
                dist = distance.euclidean(self.hidden_layer[i], self.hidden_layer[j])
                if dist > max_dist:
                    max_dist = dist
            self.scale_coefficient[i] = \
                max_dist / (numpy.sqrt(2 * numpy.size(self.hidden_layer, 0)))

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    # Jak wiadomo bias to przesunięcie wyniku o stałą więc jeżeli wybraliśmy
    # że bias istnieje to on jest po prostu dodawany do odpowiedniego wyniku iloczynu skalarnego
    # bias istnieje tylko dla output layer aktualnie
    def calculate_outputs(self, inputs):
        hidden_layer_output = []
        for i in range(numpy.size(self.hidden_layer, 0)):
            # ze wzoru, prezentacja 6, koło 20 strony, wynik dla warstwy radialnej
            dist = (distance.euclidean(self.hidden_layer[i], inputs) ** 2)
            denominator = 2 * (self.scale_coefficient[i] ** 2)
            value = numpy.exp(-1 * (dist / denominator))
            hidden_layer_output.append(value)
        # wynik dla warstwy wyjsciowej
        output_layer_output = numpy.dot(hidden_layer_output, self.output_layer.T) + self.bias_output_layer
        return hidden_layer_output, output_layer_output

    # trening, tyle razy ile podamy epochów
    # dla każdego epochu shufflujemy nasze macierze i przechodzimy przez nie po każdym wierszu z osobna
    def train(self, epoch_count):
        error_list = []
        for it in range(epoch_count):

            # Shuffle once each iteration
            joined_arrays = numpy.vstack((self.input_data, self.expected_outputs)).T
            numpy.random.shuffle(joined_arrays)
            joined_arrays_left, joined_arrays_right = numpy.hsplit(joined_arrays, 2)
            mean_squared_error = 0
            ite = 0

            for k, j in zip(joined_arrays_left, joined_arrays_right):
                ite += 1
                # epoka zwraca błąd
                mean_squared_error += self.epoch(k, j)

            mean_squared_error = mean_squared_error / ite
            error_list.append(mean_squared_error)
        print("OSTATNI BLAD", error_list[-1])
        # po przejściu przez wszystkie epoki zapisujemy błędy średniokwadratowe do pliku
        # with open("mean_squared_error.txt", "w") as file:
        with open("mean_squared_error.txt", "w") as file:
            for i in error_list:
                file.write(str(i) + "\n")

    def epoch(self, k, j):
        join_k_j = numpy.concatenate((k, j), axis=None)
        # print(join_k_j)
        hidden_layer_output, output_layer_output = self.calculate_outputs(k)
        # błąd dla wyjścia to różnica pomiędzy oczekiwanym wynikiem a otrzymanym
        output_error = output_layer_output - j
        mean_squared_error = output_error.dot(output_error) / 2

        # output_delta - współczynnik zmiany wagi dla warstwy wyjściowej. Otrzymujemy jeden współczynnik dla każdego neronu.
        # aby potem wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
        # Pochodna funkcji liniowej = 1
        output_delta = output_error * 1

        output_layer_adjustment = []

        for i in output_delta:
            value = [i * j for j in hidden_layer_output]
            output_layer_adjustment.append(value)
        output_layer_adjustment = numpy.asarray(output_layer_adjustment)

        # jeżeli wybraliśmy żeby istniał bias to teraz go modyfikujemy
        if self.is_bias:
            output_bias_adjustment = eta * output_delta + alfa * self.bias_output_layer_delta
            self.bias_output_layer -= output_bias_adjustment
            self.bias_output_layer_delta = output_bias_adjustment

        output_layer_adjustment = eta * output_layer_adjustment + alfa * self.delta_weights_output_layer

        # modyfikujemy wagi w warstwach
        self.output_layer -= output_layer_adjustment

        # zapisujemy zmianę wag by użyć ją w momentum
        self.delta_weights_output_layer = output_layer_adjustment

        return mean_squared_error


# otwieramy plik errorów i go plotujemy
def plot_file():
    with open("mean_squared_error.txt", "r") as file:
        lines = file.read().splitlines()
    values = []
    for i in lines:
        values.append(float(i))
    plt.plot(values, markersize=1)
    plt.xlabel('Epoch')
    plt.ylabel('Error for epoch')
    plt.title("Mean square error change")
    plt.show()


def plot_function(siec, title, neurons, points=None):
    if points is not None:
        values = []
        for i in points:
            values.append(siec.calculate_outputs(i[0])[1])
        plt.plot(points[:, 0], points[:, 1])
        points = points[:, 0]
        plt.plot(points, values, 'o', markersize=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("File: " + title[:-4] + ", num of neurons = " + str(neurons))
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
    numpy.random.seed(0)
    neurons = 8
    train_file = "approximation_train_1.txt"
    test_file = "approximation_test.txt"
    # ilość neuronów, ilość wyjść, ilość wejść, czy_bias
    siec = NeuralNetwork(neurons, 1, True, read_2d_float_array_from_file(train_file)[:, 0],
                         read_2d_float_array_from_file(train_file)[:, 1])
    iterations = 100
    # dane wejściowe, dane wyjściowe, ilość epochów
    siec.train(iterations)
    plot_file()
    values = read_2d_float_array_from_file(test_file)
    values2 = numpy.zeros_like(values)
    indexes = numpy.argsort(values[:, 0])
    for i in range(len(indexes)):
        values2[i] = values[indexes[i]]
    plot_function(siec, train_file, neurons, values2)


if __name__ == "__main__":
    main()
