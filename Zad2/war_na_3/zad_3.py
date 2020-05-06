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
#   dodać drugi tryb działania (Gauss)
#   dodać drugi algorytm (gaz neuronowy)
#   dodać błąd kwantyzacji
#   dodać zmianę parametru alfa
#   przeczytać uważnie całą prezentację
#   promień sąsiedztwa też ma maleć wraz z postępowaniem adaptacji


class Kohonen:
    def __init__(self, input_matrix, neuron_num, is_gauss, alfa=0.1, distance_still_change=1):
        self.neuron_num = neuron_num
        self.input_matrix = input_matrix
        self.is_gauss = is_gauss
        self.epoch_nr = 0
        self.map = np.random.normal(np.mean(input_matrix), np.std(input_matrix),
                                    size=(self.neuron_num, len(input_matrix[0])))
        self.distance_map = np.zeros_like(self.map)
        # poczatkowy wspolczynnik tego przesuwania XD nie wiem jak to nazwać, chce spac
        self.alfa = alfa
        # odleglosc dla tej metody, gdzie sie ma zmieniac tylko punkt i pobliskie. Trzeba zrobic to lepiej
        self.distance_still_change = distance_still_change

    # TODO: poprawic, aktualnie zmienia tylko w promieniu
    def epoch(self):
        np.random.shuffle(self.input_matrix)
        counter = 0
        for i in self.input_matrix:
            if not self.is_gauss:
                self.distance_map_fill(i)
                smallest_index = np.argmin(self.distance_map)
                for j in range(len(self.map)):
                    if distance.euclidean(self.map[j], self.map[smallest_index]) <= self.distance_still_change:
                        self.map[j] = self.map[j] + self.alfa * (i - self.map[j])
            else:
                self.distance_map_fill(i)
                smallest_index = np.argmin(self.distance_map)
                for j in range(len(self.map)):
                    self.map[j] = self.map[j] + self.alfa \
                        * self.euclidean_func(self.map[smallest_index], self.map[j]) * (i - self.map[j])
            counter += 1

            if counter % 100 == 0:
                print("Iteration in epoch nr", counter)

    def euclidean_func(self, pos_closest, pos_checked):
        # TODO: nie wiem w jakim zakresie dac ten self.distance_sill_change ktory wgl trzeba inaczej nazwac
        return np.exp(-distance.euclidean(pos_checked, pos_closest) ** 2 / (2 * (self.distance_still_change ** 2)))

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
    kohonen = Kohonen(read_2d_float_array_from_file("Danetestowe.txt", is_comma=True), 100, True)
    plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))
    kohonen.epoch()
    plot(kohonen.map, read_2d_float_array_from_file("Danetestowe.txt", is_comma=True))
    # kohonen = Kohonen(read_2d_float_array_from_file("punkty.txt", is_comma=False), 1000, True)
    # plot(kohonen.map, read_2d_float_array_from_file("punkty.txt", is_comma=False))
    # kohonen.epoch()
    # plot(kohonen.map, read_2d_float_array_from_file("punkty.txt", is_comma=False))


if __name__ == '__main__':
    main()
