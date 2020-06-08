from PIL import Image
import numpy as np
import datetime

from scipy.spatial import distance


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

    # jedna epoka
    def epoch(self):
        np.random.shuffle(self.input_matrix)
        current_percent = 0
        if not self.is_neural_gas:
            for i in self.input_matrix:
                self.change_alpha()
                self.change_neighbourhood_radius()

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
                                              * self.euclidean_func(self.map[smallest_index], self.map[j]) * (
                                                      i - map_not_sleeping[j])

                    for j in range(len(map_not_sleeping)):
                        self.map[true_index[j]] = map_not_sleeping[j]

                self.current_step += 1
                if (self.current_step * 100) / self.max_step > current_percent:
                    current_percent += 1
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
                for j in range(len(distance_ranking)):
                    map_not_sleeping[distance_ranking[j]] = map_not_sleeping[distance_ranking[j]] \
                                                            + self.current_alfa * self.neural_gass_neighbour_fun(j) * (
                                                                    i - map_not_sleeping[distance_ranking[j]])
                for j in range(len(map_not_sleeping)):
                    self.map[true_index[j]] = map_not_sleeping[j]

                self.current_step += 1
                if (self.current_step * 100) / self.max_step > current_percent:
                    current_percent += 1
                    print("Currently ", (self.current_step * 100) / self.max_step, "% done")
                    # counter = 0
                    # for i in self.potentials:
                    #     if i > self.min_potential:
                    #         counter += 1
                    # print(counter)

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
    def euclidean_func(self, pos_closest, pos_checked):
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
            # self.calculate_quantization_error()
            # print("current_quant_error = ", self.quantization_error_list[i])
            self.epoch()
        # self.calculate_quantization_error()
        # print("current_quant_error = ", self.quantization_error_list[-1])

    # obliczanie błędu kwantyzacji ze wzoru
    def calculate_quantization_error(self):
        print("*calculating quantization error*")
        __sum = 0
        counter = 0
        percent = 0
        for i in self.input_matrix:
            self.distance_map_fill(i)
            __sum += np.min(self.distance_map) ** 2
            counter += 1
            if (counter * 100) / self.num_rows_input_data > percent:
                percent += 1
                print("Currently ", (counter * 100) / self.num_rows_input_data, "% done")
        self.quantization_error_list.append(__sum / self.num_rows_input_data)
        with open("quantization_error.txt", "a") as file:
            file.write(str(__sum / self.num_rows_input_data) + "\n")
        pass

    # wypełniamy macierz w której odpowiadające indexy w self.map
    def distance_map_fill(self, point):
        distance_map_list = []
        for i in self.map:
            distance_map_list.append(distance.euclidean(i, point))
        self.distance_map = np.asarray(distance_map_list)


def image_pixels_to_array(image):
    size_x, size_y = image.size
    pix = image.load()
    pixels_list = []
    for i in range(size_x):
        for j in range(size_y):
            pixels_list.append(pix[i, j])
    return np.asarray(pixels_list)


def save_new_picture(image, kohonen):

    def distance_map(point, kohonen_map):
        distance_map_list = []
        for i in kohonen_map:
            distance_map_list.append(distance.euclidean(i, point))
        return np.asarray(distance_map_list)

    def totuple(a):
        try:
            return tuple(totuple(int(i)) for i in a)
        except TypeError:
            return a

    if type(kohonen) is KohonenOrNeuralGas:
        size_x, size_y = image.size
        pix = image.load()
        percent = 0
        for i in range(size_x):
            for j in range(size_y):
                distance_list = distance_map(pix[i, j], kohonen.map)
                index = np.argmin(distance_list)
                pix[i, j] = totuple(np.rint(kohonen.map[index]))

            if (i * 100) / size_x > percent:
                percent += 1
                print("Currently ", (i * 100) / size_x, "% done (writing to file)")
        image.save('SUPEROBRAZEK.jpg')


def main():
    im = Image.open('12-Angry-Men-The-Jurors-700x525.jpg')
    kohonen = KohonenOrNeuralGas(input_matrix=image_pixels_to_array(im),
                                 neuron_num=16,
                                 is_gauss=True, is_neural_gas=True, epoch_count=1, neighbourhood_radius=1.5,
                                 min_potential=0, alfa=0.8)

    a = datetime.datetime.now()
    kohonen.train()
    b = datetime.datetime.now()
    save_new_picture(im, kohonen)
    c = b - a
    print(c)


if __name__ == '__main__':
    main()
