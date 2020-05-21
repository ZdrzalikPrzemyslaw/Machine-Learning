import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance


# wagi - współrzędne punktu

# TODO:
#   przeczytać uważnie całą prezentację

class KohonenOrNeuralGas:
    # alfa - wpsolczynnik uczenia, neighbourhood_radius - to co we wzorach jest opisane lambda
    # dla kazdej metody to nieco inne jest ale generalnie uzywane w liczeniu tego G(i, x) co jest we wzorach

    def __init__(self, input_matrix, neuron_num, is_neural_gas=False,
                 is_gauss=True, alfa=0.6, neighbourhood_radius=0.3, epoch_count=1, min_potential=0.75):
        # liczba neuronów i dane wejsciowe
        self.neuron_num = neuron_num
        self.input_matrix = input_matrix

        # wybór podmetody (gauss) i metody (neural_gas) jesli neural gas - true to gauss nie ma znaczenia
        self.is_gauss = is_gauss
        self.is_neural_gas = is_neural_gas

        # ile epok
        self.epoch_count = epoch_count

        # losujemy startowe pozycje neuronów
        self.map = np.random.normal(np.mean(input_matrix), np.std(input_matrix),
                                    size=(self.neuron_num, len(input_matrix[0])))

        # wspolczynnik uczenia, max, min i current - zmienia sie w trakcie
        self.alfa_max = alfa
        self.alfa_min = 0.0001
        self.current_alfa = self.alfa_max

        # ten drugi wspolczynnik lambda, max, min i current - zmienia sie w trakcie
        self.neighbourhood_radius_max = neighbourhood_radius
        self.neighbourhood_radius_min = 0.0001
        self.current_neighbourhood_radius = self.neighbourhood_radius_max

        # uzywamy w 2 miejscach, generalnie srednio potrzebne
        self.num_rows_input_data, self.num_cols_input_data = self.input_matrix.shape

        # tutaj pozniej przechowujemy odleglosci odpowiednich neuronów od aktualnie rozpatrywanego wektoru wejściowego
        self.distance_map = np.zeros(neuron_num)
        # potencjaly do matwych neuronów
        # ustawiamy na 0 to mamy dzialanie jak wczesniej
        self.potentials = np.ones(neuron_num)
        self.min_potential = min_potential

        # aktualny krok i maksymalna liczba kroków (liczba rzędów w wejsciu razy liczba epok)
        # uzywamy maxymalny step i current step do liczenia tych current alfa i current_neighbourhood_radius
        self.current_step = 0
        self.max_step = self.num_rows_input_data * self.epoch_count

        # tutaj przechowujemy błędy liczone po każdej epoce (bardzo wolno liczy się błąd)
        self.quantization_error_list = []

        self.animation_list = []

    # jedna epoka
    def epoch(self):
        np.random.shuffle(self.input_matrix)
        if not self.is_neural_gas:
            for i in self.input_matrix:
                self.change_alpha()
                self.change_neighbourhood_radius()
                self.animation_list.append(np.copy(self.map))

                # klasyczny wariant Kohenena,
                # modyfikacja zwyciezcy oraz znajdujących się o self.current_neighbourhood_radius od niego neuronów
                if not self.is_gauss:
                    self.distance_map_fill(i)
                    map_not_sleeping, distance_map_not_sleeping, true_index = \
                        self.get_not_sleeping_neurons_and_distances()
                    self.change_potentials(true_index[np.argmin(distance_map_not_sleeping)])
                    smallest_index = np.argmin(distance_map_not_sleeping)
                    for j in range(len(map_not_sleeping)):
                        # sprawdzamy czy odległość neuronu od zwycięzcy jest mniejsza niż current_neighbourhood_radius
                        # jesli tak to modyfikujemy zgodnie ze wzorem
                        if distance.euclidean(map_not_sleeping[j],
                                              map_not_sleeping[smallest_index]) <= self.current_neighbourhood_radius:
                            map_not_sleeping[j] = map_not_sleeping[j] + self.current_alfa * (i - map_not_sleeping[j])
                    for j in range(len(map_not_sleeping)):
                        self.map[true_index[j]] = map_not_sleeping[j]

                # wariant gaussa
                # modyfikacja zwycięzcy oraz wszystkich innych w zależności od ich odległości od zwycięzcy
                else:
                    self.distance_map_fill(i)
                    map_not_sleeping, distance_map_not_sleeping, true_index = \
                        self.get_not_sleeping_neurons_and_distances()
                    self.change_potentials(true_index[np.argmin(distance_map_not_sleeping)])
                    smallest_index = np.argmin(distance_map_not_sleeping)
                    for j in range(len(map_not_sleeping)):
                        map_not_sleeping[j] = map_not_sleeping[j] + self.current_alfa \
                                              * self.gauss_neighbourhood_function(self.map[smallest_index], self.map[j]) * (
                                                          i - map_not_sleeping[j])

                    for j in range(len(map_not_sleeping)):
                        self.map[true_index[j]] = map_not_sleeping[j]

                self.current_step += 1
                if self.current_step % 100 == 0:
                    print("Currently ", (self.current_step * 100) / self.max_step, "% done")

        # metoda gazu neuronowego
        # sortujemy neurony wg odległości od aktualnego wektoru wejścia
        # liczymy zmianę pozycji w zależności od pozycji w rankingu a nie od faktycznej odległosci
        else:
            for i in self.input_matrix:
                self.change_alpha()
                self.change_neighbourhood_radius()
                self.distance_map_fill(i)
                map_not_sleeping, distance_map_not_sleeping, true_index = self.get_not_sleeping_neurons_and_distances()
                distance_ranking = np.argsort(distance_map_not_sleeping)
                self.change_potentials(true_index[np.argmin(distance_map_not_sleeping)])
                self.animation_list.append(np.copy(self.map))
                for j in range(len(distance_ranking)):
                    map_not_sleeping[distance_ranking[j]] = map_not_sleeping[distance_ranking[j]] \
                                                            + self.current_alfa * self.neural_gass_neighbour_fun(j) * (
                                                                    i - map_not_sleeping[distance_ranking[j]])
                for j in range(len(map_not_sleeping)):
                    self.map[true_index[j]] = map_not_sleeping[j]

                self.current_step += 1
                if self.current_step % 100 == 0:
                    print("Currently ", (self.current_step * 100) / self.max_step, "% done")
                    # counter = 0
                    # for i in self.potentials:
                    #     if i > self.min_potential:
                    #         counter += 1
                    # print(counter)

        self.animation_list.append(np.copy(self.map))

    # zmiana potencjałów dla
    def change_potentials(self, index):
        self.potentials += 1 / len(self.potentials)
        self.potentials[index] -= 1 / len(self.potentials)
        self.potentials[index] -= self.min_potential

    def get_not_sleeping_neurons_and_distances(self):
        neuron_list = []
        distance_list = []
        true_index_list = []
        for i in range(len(self.map)):
            if self.potentials[i] >= self.min_potential:
                neuron_list.append(self.map[i])
                distance_list.append(self.distance_map[i])
                true_index_list.append(i)
        return np.asarray(neuron_list), np.asarray(distance_list), np.asarray(true_index_list)

    # dla gazu neuronowego zwraca współczynnik związany z rankingiem punktu
    def neural_gass_neighbour_fun(self, ranking):
        return np.exp(-ranking / self.current_neighbourhood_radius)

    # funkcja okreslajaca wspolczynnik zwiazany z odleglością punktów od zwycieskiego
    # dla metody euklidesowej w Kohonenie
    def gauss_neighbourhood_function(self, pos_closest, pos_checked):
        return np.exp(
            -distance.euclidean(pos_checked, pos_closest) ** 2 / (2 * (self.current_neighbourhood_radius ** 2)))

    # zmiana współczynnika lambda
    def change_neighbourhood_radius(self):
        self.current_neighbourhood_radius = self.neighbourhood_radius_max \
                                            * (self.neighbourhood_radius_min / self.neighbourhood_radius_max) \
                                            ** (self.current_step / self.max_step)

    # zmiana współczynnika alfa
    def change_alpha(self):
        self.current_alfa = self.alfa_max * (self.alfa_min / self.alfa_max) ** (self.current_step / self.max_step)

    # nauka + liczymy błędy kwantyzacji
    def train(self):
        for i in range(self.epoch_count):
            self.calculate_quantization_error()
            print("current_quant_error = ", self.quantization_error_list[i])
            self.epoch()
        self.calculate_quantization_error()
        print("current_quant_error = ", self.quantization_error_list[-1])

    # obliczanie błędu kwantyzacji ze wzoru
    def calculate_quantization_error(self):
        print("*calculating quantization error*")
        __sum = 0
        for i in self.input_matrix:
            self.distance_map_fill(i)
            __sum += np.min(self.distance_map) ** 2
        self.quantization_error_list.append(__sum / self.num_rows_input_data)

    # wypełniamy macierz w której odpowiadające indexy w self.map
    def distance_map_fill(self, point):
        distance_map_list = []
        for i in self.map:
            distance_map_list.append(distance.euclidean(i, point))
        self.distance_map = np.asarray(distance_map_list)

    def animate_training(self):
        fig, ax = plt.subplots()

        ax.axis([np.min(self.animation_list[0], axis=0)[0] - 1, np.max(self.animation_list[0], axis=0)[0] + 1,
                 np.min(self.animation_list[0], axis=0)[1] - 1, np.max(self.animation_list[0], axis=0)[1] + 1])

        ax.plot(self.input_matrix[:, 0], self.input_matrix[:, 1], 'ro')
        l, = ax.plot([], [], 'bo')

        def animate(i):
            if i > len(self.animation_list) - 1:
                i = len(self.animation_list) - 1
            l.set_data(self.animation_list[i][:, 0], self.animation_list[i][:, 1])
            ax.set_title("Step nr " + str(i))
            return l

        ani = animation.FuncAnimation(fig, animate, interval=1, repeat=False)
        plt.show()


def read_2d_float_array_from_file(file_name, is_comma=False):
    two_dim_list_of_return_values = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        if not is_comma:
            for j in list(map(float, i.split())):
                one_dim_list.append(j)
            two_dim_list_of_return_values.append(one_dim_list)
        else:
            for j in list(map(float, i.split(","))):
                one_dim_list.append(j)
            two_dim_list_of_return_values.append(one_dim_list)
    return np.asarray(two_dim_list_of_return_values)


def plot(list2d, list2d2=None):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    if list2d2 is not None:
        for i in list2d2:
            list3.append(i[0])
            list4.append(i[1])
        plt.plot(list3, list4, 'bo', color='red')
    for i in list2d:
        list1.append(i[0])
        list2.append(i[1])
    plt.plot(list1, list2, 'bo')
    plt.show()


def main():
    kohonen = KohonenOrNeuralGas(input_matrix=read_2d_float_array_from_file("Danetestowe.txt", is_comma=True),
                                 neuron_num=100,
                                 is_gauss=False, is_neural_gas=False, epoch_count=1, neighbourhood_radius=0.5,
                                 min_potential=0.75, alfa=1.5)
    # plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))
    kohonen.train()
    # plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))

    kohonen.animate_training()


if __name__ == '__main__':
    main()
