import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.centr = 0
        self.color = ''
        
#zrodlo: https://www.statystyka.az.pl/analiza-skupien/metoda-k-srednich.php

class KMean:
    def __init__(self, input_matrix, points_matrix, number_of_centroids, number_of_epoch):
        #input matrix [x, y, centroid]
        self.input_matrix = input_matrix
        #points matrix
        self.points = points_matrix
        #liczba epok
        self.number_of_epoch = number_of_epoch
        #liczba centroidow
        self.number_of_centroids = number_of_centroids
        #losujemy startowe pozycje centroidow
        self.centroid = np.random.normal(np.mean(input_matrix), np.std(input_matrix), 
                                        size=(self.number_of_centroids, len(input_matrix[0])-1))
        #odleglosc punktow od poszczegolnych centroidow
        self.distance_to_centr = np.zeros(len(input_matrix))
        self.distance_to_centr_object = np.zeros(len(points_matrix))

        #animacja
        self.animation_list = []
        self.points_animation = []
        self.test = []
        
    
    #obliczany odleglosc punktow od poszczegolnych centroidow
    def calculate_distance(self):
        #usuwamy kolumne przynaleznosci tak aby zostaly nam same wspolrzedne
        temp_points = np.delete(self.input_matrix, 2, 1)
        test = []
        for i in temp_points:
            point_to_centr = []
            for j in self.centroid:
                point_to_centr.append(distance.euclidean(i,j))
            test.append(point_to_centr)
        self.distance_to_centr = np.asarray(test)
    
    #funkcja tworzaca array wspolrzednych
    def points_to_array(self):
        points = []
        for i in self.points:
            individual = []
            individual.append(i.x)
            individual.append(i.y)
            points.append(individual)
        return np.asarray(points)

    #obliczamy odleglosc punktow od poszczegolnych centroidow
    def distance_calc(self):
        test = []
        point_array = self.points_to_array()
        for i in point_array:
            point_to_centre = []
            for j in self.centroid:
                point_to_centre.append(distance.euclidean(i,j))
            test.append(point_to_centre)
        self.distance_to_centr_object = np.asarray(test)
    #okreslamy przynaleznosc punktow do danych centroidow na podstawie odleglosci
    def points_to_centr(self):
        j=0
        for i in self.distance_to_centr:
            for k in range (len(i)):
                if (i.min() == i[k]):
                    self.input_matrix[j,2] = k+1
            j+=1
    def points_to_centr_object(self):
        j=0
        color = ['c', 'm', 'g', 'r', 'b']
        for i in self.distance_to_centr_object:
            for k in range (len(i)):
                if (i.min() == i[k]):
                    self.points[j].centr = k+1
                    self.points[j].color = color[k]
            j+=1
    #ustawiamy nowe pozycje centroidow na podstawie sredniej arytmetycznej pozycji
    #punktow przypisanych do danego centroidu
    def set_new_centre_position(self):
        for i in range(len(self.centroid)):
            counter = 0
            new_position = []
            sumx = 0
            sumy = 0
            for point in self.points:
                if (point.centr == i+1):
                    sumx += point.x
                    sumy += point.y
                    counter += 1
            new_position.append(sumx/counter)
            new_position.append(sumy/counter)
            self.centroid[i] = new_position
    #glowny algorytm
    def train(self):
        # 1. Ustalamy liczbe skupien -> parametr konstruktora
        # 2. Ustalamy wstepne srodki skupien (centroidy) -> konstruktor
        # 3. 4. 5. powtarzamy w zaleznosci od liczby epok
        for i in range(self.number_of_epoch):
            #debug xD
            # print(self.centroid)
            # print("\n")
            
            # 3. Obliczamy odleglosc obiektow od srodkow skupien
            self.distance_calc()

            # 4. Przypisujemy obiekty do skupien
            self.points_to_centr_object()
            # 5. Ustalamy nowe pozycje centroidow
            self.animate_plot()
            self.animation_list.append(np.copy(self.centroid))
            self.points_animation.append(np.copy(self.points))
            # for i in self.points_animation:
            #     for j in i:
            #         plt.scatter(j.x, j.y, c=j.color)

            # plt.show()
            self.set_new_centre_position()

    def animate_plot(self):
        px1 = []
        for point in self.points:
            test = []
            test.append(point.x)
            test.append(point.y)
            test.append(point.color)
            px1.append(test)
        test2 = np.asarray(px1)
        self.test.append(np.asarray(px1))

    def animate_training(self):
        fig, ax = plt.subplots()

        # ax.axis([np.min(self.animation_list[0], axis=0)[0] - 1, np.max(self.animation_list[0], axis=0)[0] + 1,
        #          np.min(self.animation_list[0], axis=0)[1] - 1, np.max(self.animation_list[0], axis=0)[1] + 1])
        ax.axis([-10, 10,
                 -10, 10])
        for eksde in self.points_animation:
                for zarazjebne in eksde:
                    ax.scatter(zarazjebne.x, zarazjebne.y, c=zarazjebne.color) 
             
        l, = ax.plot([], [],'ko')

        def animate(i):
            if i > len(self.animation_list) - 1:
                i = len(self.animation_list) - 1
              
            l.set_data(self.animation_list[i][:,0], self.animation_list[i][:,1])
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
                #dodajemy 0 jako kolumne przynaleznosci do danego centroidu
                one_dim_list.append(0)
                two_dim_list_of_return_values.append(one_dim_list)
                
        return np.asarray(two_dim_list_of_return_values)

#tworzymy array punktow z pliku
def create_points(file_name, is_comma=False):
    points_array = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        test = []
        if not is_comma:
            return
        else:
            for j in list(map(float, i.split(","))):
                test.append(j)
        points_array.append(Point(test[0],test[1]))
    return np.asarray(points_array)

def main():

    NUMBER_OF_CENTROIDS = 2
    NUMBER_OF_EPOCH = 10


    kMean = KMean(input_matrix=read_2d_float_array_from_file("Zad2/war_na_3/Danetestowe.txt",is_comma=True),
                 points_matrix=create_points("Zad2/war_na_3/test.txt", is_comma=True), 
                number_of_centroids=NUMBER_OF_CENTROIDS, number_of_epoch=NUMBER_OF_EPOCH)
    
    
    kMean.train()
    kMean.animate_training()
    print("Hello")

if __name__ == '__main__':
    main()