from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance

# wagi - współrzędne punktu

"""
pomysl:
- zainicjalizować klasę, o N punktach
- ustawić ładnie punkty w przestrzeni 2D żeby slicznie było
- każdy epoch:
    losujemy punkty kolejność
    przechodizmy przez kolejne punkty
    liczymy odległość każdego punktu w przestrzeni od aktualnie rozpatrywanego punktu
    szukamy Nearest
    modyfikujemy pozycję punktów w zależności od algorytmu
    nie wiem czy ten współczynnik modyfikacji ma się zmieniać co cały epoch czy przy każdym punkcie? 
    ale chyba co caly epoch
    no i w sumie tyle XD 
"""


# TODO:
#   Pokonać problem martwych neuronów
#   dodać błąd kwantyzacji
#   przeczytać uważnie całą prezentację


class Kohonen:
    def __init__(self, input_matrix, neuron_num, is_neural_gas=False, is_gauss=True, alfa=0.6, neighbourhood_radius=0.5, epoch_count=1):
        self.neuron_num = neuron_num
        self.input_matrix = input_matrix
        self.is_gauss = is_gauss
        self.epoch_nr = 0
        self.epoch_count = epoch_count
        self.map = np.random.normal(np.mean(input_matrix), np.std(input_matrix),
                                    size=(self.neuron_num, len(input_matrix[0])))
        self.distance_map = np.zeros_like(self.map)
        # poczatkowy wspolczynnik tego przesuwania XD nie wiem jak to nazwać, chce spac
        self.max_alfa = alfa
        self.min_alfa = 0.01
        self.current_alfa = self.max_alfa
        # Odleglosc od punktu zwycięskiego lub mianownik w drugiej metodzie
        self.neighbourhood_radius_max = neighbourhood_radius
        self.current_neighbourhood_radius = self.neighbourhood_radius_max
        self.neighbourhood_radius_min = 0.01
        self.current_step = 1
        self.max_step = len(self.input_matrix) / len(self.input_matrix[0]) * self.epoch_count
        self.is_neural_gas = is_neural_gas

    # TODO: poprawic, aktualnie zmienia tylko w promieniu
    def epoch(self):
        np.random.shuffle(self.input_matrix)
        if not self.is_neural_gas:
            for i in self.input_matrix:
                self.change_alpha()
                self.change_neighbourhood_radius()
                if not self.is_gauss:
                    self.distance_map_fill(i)
                    smallest_index = np.argmin(self.distance_map)
                    for j in range(len(self.map)):
                        if distance.euclidean(self.map[j], self.map[smallest_index]) <= self.current_neighbourhood_radius:
                            self.map[j] = self.map[j] + self.current_alfa * (i - self.map[j])

                else:
                    self.distance_map_fill(i)
                    smallest_index = np.argmin(self.distance_map)
                    for j in range(len(self.map)):
                        self.map[j] = self.map[j] + self.current_alfa \
                                      * self.euclidean_func(self.map[smallest_index], self.map[j]) * (i - self.map[j])

                self.current_step += 1
                if self.current_step % 100 == 0:
                    print("Iteration in epoch nr", self.current_step)
                    print("Current alfa", self.current_alfa)
                    print("Current neighbour radius:", self.current_neighbourhood_radius)

        else:
            for i in self.input_matrix:
                self.change_alpha()
                self.change_neighbourhood_radius()
                self.distance_map_fill(i)
                distance_ranking = np.argsort(self.distance_map)
                for j in range(len(distance_ranking)):
                    self.map[distance_ranking[j]] = self.map[distance_ranking[j]] \
                            + self.current_alfa * self.neural_gass_neighbour_fun(j) * (i - self.map[distance_ranking[j]])

                self.current_step += 1
                if self.current_step % 100 == 0:
                    print("Iteration in epoch nr", self.current_step)
                    print("Current alfa", self.current_alfa)
                    print("Current neighbour radius:", self.current_neighbourhood_radius)

    def neural_gass_neighbour_fun(self, ranking):
        return np.exp(-ranking / self.current_neighbourhood_radius)

    def euclidean_func(self, pos_closest, pos_checked):
        return np.exp(
            -distance.euclidean(pos_checked, pos_closest) ** 2 / (2 * (self.current_neighbourhood_radius ** 2)))

    def change_neighbourhood_radius(self):
        self.current_neighbourhood_radius = self.neighbourhood_radius_max \
                                            * (self.neighbourhood_radius_min / self.neighbourhood_radius_max) \
                                            ** (self.current_step / self.max_step)

    def change_alpha(self):
        self.current_alfa = self.max_alfa * (self.min_alfa / self.max_alfa) ** (self.current_step / self.max_step)

    def train(self):
        for i in range(self.epoch_count):
            self.epoch()

    def distance_map_fill(self, point):
        # TODO zmien nazwe zmiennej
        potezna_lista = []
        for i in self.map:
            potezna_lista.append(distance.euclidean(i, point))
        self.distance_map = np.asarray(potezna_lista)


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
    if not list2d2 is None:
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
    kohonen = Kohonen(input_matrix=read_2d_float_array_from_file("Danetestowe.txt", is_comma=True),
                      neuron_num=200, is_neural_gas=True, epoch_count=1)
    plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))
    kohonen.train()
    plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))
    # kohonen = Kohonen(read_2d_float_array_from_file("punkty.txt", is_comma=False), 1000, True)
    # plot(kohonen.map, read_2d_float_array_from_file("punkty.txt", is_comma=False))
    # kohonen.epoch()
    # plot(kohonen.map, read_2d_float_array_from_file("punkty.txt", is_comma=False))


if __name__ == '__main__':
    main()
