import numpy
import time

import matplotlib.pyplot as plt

# wspolczynnik uczenia
eta = 0.1
# momentum
alfa = 0.7


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
        # wagi inicjalizujemy liczbami w przedziale -1 do 1
        self.hidden_layer = (2 * numpy.random.random((number_of_inputs, number_of_neurons_hidden_layer)).T - 1)
        self.delta_weights_hidden_layer = numpy.zeros((number_of_inputs, number_of_neurons_hidden_layer)).T
        self.output_layer = 2 * numpy.random.random((number_of_neurons_hidden_layer, number_of_neurons_output)).T - 1
        self.delta_weights_output_layer = numpy.zeros((number_of_neurons_hidden_layer, number_of_neurons_output)).T
        # jesli wybralismy że bias ma byc to tworzymy dla każdej warstwy wektor wag biasu
        # 1 bias na 1 neuron w każdej warstwie
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

    # Funckja sigmoidalna
    def sigmoid_fun(self, inputcik):
        return 1 / (1 + numpy.exp(-inputcik))

    # interesujące jest to, że według mojej wiedzy te wzory są równe sobie a dają dość bardzo różne wyniki w niektórych przypadkach
    # z wolfram alpha
    # def sigmoid_fun_deriative(self, inputcik):
    #     return numpy.exp(-inputcik) /  ((numpy.exp(-inputcik) + 1) ** 2)

    # pochodna funkcji sigmoidalnej
    def sigmoid_fun_deriative(self, inputcik):
        return inputcik * (1 - inputcik)

    # najpierw liczymy wynik z warstwy ukrytej i potem korzystając z niego liczymy wynik dla neuronów wyjścia
    # Jak wiadomo bias to przesunięcie wyniku o stałą więc jeżeli wybraliśmy że bias istnieje to on jest po prostu dodawany do odpowiedniego wyniku iloczynu skalarnego
    def calculate_outputs(self, inputs):
        # wynik dla neuronu to suma wyników wejść i odpowiednich wag + bias tego neuronu i od tego liczymy wartość funkcji aktywacji
        hidden_layer_output = self.sigmoid_fun(numpy.dot(inputs, self.hidden_layer.T) + self.bias_hidden_layer)
        output_layer_output = self.sigmoid_fun(
            numpy.dot(hidden_layer_output, self.output_layer.T) + self.bias_output_layer)
        return hidden_layer_output, output_layer_output

    # trening, tyle razy ile podamy epochów
    # dla każdego epochu shufflujemy nasze macierze i przechodzimy przez nie po każdym wierszu z osobna
    def train(self, inputs, expected_outputs, epoch_count):
        # tutaj sa wszystkie błędy w każdej iteracji
        error_list = []
        # lista tego ile w danej iteracji przyporządkowaliśmy dobrze:
        # wszystkiego, klasy 0, klasy 1, klasy 2.
        correct_list = []
        correct_0_list = []
        correct_1_list = []
        correct_2_list = []
        # łaczymy wejścia i oczekiwane wyjścia żeby móc zrobić ładne losowanie
        joined_arrays = numpy.vstack((inputs.T, expected_outputs.T)).T
        for it in range(epoch_count):
            # Losujemy kolejność
            numpy.random.shuffle(joined_arrays)
            # dzielimy na 2 macierze, wejść i oczekiwanych wyjść
            joined_arrays_left = joined_arrays[:, :-1]
            joined_arrays_right = joined_arrays[:, -1:]
            # ile danej klasy było ogólnie w przejściu
            correct_0_amount = 0
            correct_1_amount = 0
            correct_2_amount = 0
            # ile dobrze przy danej iteracji
            correct_all = 0
            correct_0 = 0
            correct_1 = 0
            correct_2 = 0
            # błąd średniokwadratowy
            mean_squared_error = 0
            # iteracja w danej epoce
            ite = 0

            for input_data, expected_outputs in zip(joined_arrays_left, joined_arrays_right):

                # wyliczamy wyjścia z obu warstw
                hidden_layer_output, output_layer_output = self.calculate_outputs(input_data)

                # SPAGHETTI CODE HERE
                # generalnie to można było zrobić tyle razy lepiej ale jest tak
                # sieć ma 3 wyjscia,
                # tam gdzie otrzymamy największą wartość uznajemy że tak sieć zklasyfikowała obiekt
                # klasy mamy podane jako liczbę (1, 2, 3)
                # w bardzo karkołomny sposób przechodzimy z tej liczby na '1'
                # na odpowiadającym temu co ma wyjśc z sieci miejscu
                # tzn jeśli klasa jest 3, to sieć wyprodukuje największa wartość na wyjściu nr 3
                # więc otrzymamy wektor wyników [~0, ~0, ~1]
                # więc by policzyć odpowiedni błąd oczekiwane wyniki muszą być w formacie
                # [0, 0, 1] i tak jest.
                old_j = expected_outputs
                if expected_outputs == 1:
                    expected_outputs = numpy.asarray([1, 0, 0])
                    correct_0_amount += 1
                elif expected_outputs == 2:
                    expected_outputs = numpy.asarray([0, 1, 0])
                    correct_1_amount += 1
                elif expected_outputs == 3:
                    expected_outputs = numpy.asarray([0, 0, 1])
                    correct_2_amount += 1
                if numpy.argmax(expected_outputs, axis=0) == numpy.argmax(output_layer_output, axis=0):
                    correct_all += 1
                    if old_j == 1:
                        correct_0 += 1
                    if old_j == 2:
                        correct_1 += 1
                    if old_j == 3:
                        correct_2 += 1
                # błąd dla wyjścia to różnica pomiędzy oczekiwanym wynikiem a otrzymanym
                output_error = output_layer_output - expected_outputs
                # błąd średniokwadratowy
                mean_squared_error += output_error.dot(output_error) / 2
                ite += 1

                # output_delta - współczynnik zmiany wagi dla warstwy wyjściowej. Otrzymujemy jeden współczynnik dla każdego neronu.
                # aby potem wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                # Pochodna funkcji liniowej = 1
                output_delta = output_error * self.sigmoid_fun_deriative(output_layer_output)

                # korzystamy z wcześniej otrzymanego współczynniku błędu aby wyznaczyć błąd dla warstwy ukrytej
                hidden_layer_error = output_delta.T.dot(self.output_layer)
                # jak dla warstwy wyjściowej hidden_layer_delta jest jeden dla każdego neuronu i
                # aby wyznaczyć zmianę wag przemnażamy go przez input odpowiadający wadze neuronu
                hidden_layer_delta = hidden_layer_error * self.sigmoid_fun_deriative(hidden_layer_output)

                # wyznaczamy obliczony 'współczynnik' przez odpowiednie wagi w warstwach by wyznaczyć zmianę dla danej wagi
                output_layer_adjustment = []
                for i in output_delta:
                    output_layer_adjustment.append(hidden_layer_output * i)
                output_layer_adjustment = numpy.asarray(output_layer_adjustment)

                hidden_layer_adjustment = []
                for i in hidden_layer_delta:
                    hidden_layer_adjustment.append(input_data * i)
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
            correct_list.append(correct_all / ite)
            correct_0_list.append(correct_0 / correct_0_amount)
            correct_1_list.append(correct_1 / correct_1_amount)
            correct_2_list.append(correct_2 / correct_2_amount)
        print("OSTATNI BLAD", error_list[-1])
        # po przejściu przez wszystkie epoki zapisujemy błędy średniokwadratowe do pliku
        with open("mean_squared_error.txt", "w") as file:
            for i in error_list:
                file.write(str(i) + "\n")
        with open("correct_assigment.txt", "w") as file:
            for i in correct_list:
                file.write(str(i) + "\n")
        with open("correct_assigment0.txt", "w") as file:
            for i in correct_0_list:
                file.write(str(i) + "\n")
        with open("correct_assigment1.txt", "w") as file:
            for i in correct_1_list:
                file.write(str(i) + "\n")
        with open("correct_assigment2.txt", "w") as file:
            for i in correct_2_list:
                file.write(str(i) + "\n")


# otwieramy plik errorów i go plotujemy
def plot_file(name="mean_squared_error.txt"):
    if name == "correct_assigment.txt":
        with open("correct_assigment0.txt", "r") as file:
            lines = file.read().splitlines()
        values0 = []
        for i in lines:
            values0.append(float(i))
        with open("correct_assigment1.txt", "r") as file:
            lines = file.read().splitlines()
        values1 = []
        for i in lines:
            values1.append(float(i))
        with open("correct_assigment2.txt", "r") as file:
            lines = file.read().splitlines()
        values2 = []
        for i in lines:
            values2.append(float(i))
        with open(name, "r") as file:
            lines = file.read().splitlines()
        values = []
        for i in lines:
            values.append(float(i))
        plt.xlabel('Epoch')
        plt.ylabel('Correct Assigments')
        plt.plot(values, 'o', markersize=2, label="All classes")
        plt.plot(values0, 'o', markersize=2, label="Class 1")
        plt.plot(values1, 'o', markersize=2, label="Class 2")
        plt.plot(values2, 'o', markersize=2, label="Class 3")
        plt.legend()
    else:
        with open(name, "r") as file:
            lines = file.read().splitlines()
        values = []
        for i in lines:
            values.append(float(i))
        plt.xlabel('Epoch')
        plt.ylabel('Error for Epoch')
        plt.title("Mean square error change")
        plt.plot(values, markersize=1)
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

    # neurony w warstwie ukrytej, wyjścia, wejścia, bias
    neurons = 5
    siec = NeuralNetwork(number_of_neurons_hidden_layer=neurons, is_bias=True,
                         number_of_neurons_output=3, number_of_inputs=1)
    train_file = "classification_train.txt"
    num_of_iterations = 100
    siec.train(numpy.delete(read_2d_float_array_from_file(train_file), [0, 1, 2], 1)[:, :-1],
               read_2d_float_array_from_file(train_file)[:, -1:], num_of_iterations)
    plot_file()
    plot_file("correct_assigment.txt")
    # sprawdzenie dla zbioru testowego
    # correct_amount = 0
    # all_1 = [0, 0, 0]
    # all_2 = [0, 0, 0]
    # all_3 = [0, 0, 0]
    # it = 0
    # for i in read_2d_float_array_from_file("classification_test.txt")[:, :]:
    #     obliczone = numpy.argmax(siec.calculate_outputs(i[:-1])[1], axis=0)
    #     if i[-1:] == 1:
    #         all_1[obliczone] += 1
    #     elif i[-1:] == 2:
    #         all_2[obliczone] += 1
    #     elif i[-1:] == 3:
    #         all_3[obliczone] += 1
    #     if numpy.argmax(siec.calculate_outputs(i[:-1])[1], axis=0) == i[-1:] - 1:
    #         correct_amount += 1
    #     it += 1
    # print("KLASYFIKACJA OBIEKTOW  :   1,  2,  3")
    # print("KLASYFIKACJA OBIEKTU 1 : ", all_1)
    # print("KLASYFIKACJA OBIEKTU 2 : ", all_2)
    # print("KLASYFIKACJA OBIEKTU 3 : ", all_3)
    # print("ILOŚC Wszystkich: ", it)
    # print("ILOŚć Odgadnietych: ", correct_amount)


if __name__ == "__main__":
    main()
