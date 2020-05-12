import numpy as np
import matplotlib
import random
import math
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.minimal_distance = 0

    #obliczamy odleglosc euklidesowa od poszczegolnych centrow
    def calculate_distance(self, center):
        point_vector = []
        center_vector = []
        point_vector.append(self.x)
        point_vector.append(self.y)
        center_vector.append(center.x)
        center_vector.append(center.y)
        return distance.euclidean(point_vector, center_vector)

    #wybieramy najmniejsza odleglosc
    def closest_center(self, centers):
        dist = []
        for center in centers:
            dist.append(self.calculate_distance(center))
        self.minimal_distance = min(dist)

class Center:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.points_assigned = []
        self.quantization_error = 0

    def assign_points(self, points):
        self.points_assigned.append(points)

    def clear_assigned_points(self):
        self.points_assigned.clear()

    #funkcja oblicza nowa wspolrzedna srodka skupiska jako srednia arytmetyczna 
    #wspolrzednych punktow nalezacych do danego skupiska
    def calculate_new_position(self):
        if (len(self.points_assigned) == 0):
            return
        
        sum_x = 0
        sum_y = 0

        for point in self.points_assigned:
            sum_x += point.x
            sum_y += point.y

        self.x = sum_x/len(self.points_assigned)
        self.y = sum_y/len(self.points_assigned)

    def calculate_error(self):
        sum = 0
        for point in self.points_assigned:
            sum += math.pow(point.calculate_distance(self),2)
        self.quantization_error = sum

#zrodlo: https://www.statystyka.az.pl/analiza-skupien/metoda-k-srednich.php

class KMean:
    def __init__(self, points_matrix, number_of_centroids, number_of_epoch):
        #points matrix
        self.points = points_matrix
        #liczba epok
        self.number_of_epoch = number_of_epoch
        #bool do warunku stopu
        if self.number_of_epoch > 0:
            self.epoch_limit = True
        else:
            self.epoch_limit = False
        #liczba centroidow
        self.number_of_centroids = number_of_centroids
        #losujemy startowe pozycje centroidow
        self.centers = self.create_centers()
        #lista obrazow do animacji
        self.images = []
        #counter epok
        self.epoch_counter = 0
        #czy centra ustabilizowane (warunek stopu)
        self.is_not_stabilized = True
        #Blad
        self.quantization_error = 0
        #vector do wykresu bledu
        self.plot_error = [[], []]
        
    #funkcja tworzaca nowe centra o losowych wspolrzednych w zakresie zaleznym od wspolrzednych punktow
    def create_centers(self):
        points_x = []
        points_y = []
        centers = []
        for point in self.points:
            points_x.append(point.x)
            points_y. append(point.y)
        for i in range(self.number_of_centroids):
            center = generate_random_points_in_range(math.floor(min(points_x)), math.ceil(max(points_x)),
                                                    math.floor(min(points_y)), math.ceil(max(points_y)))
            centers.append(Center(center[0], center[1]))
        return np.asarray(centers)
    
    #funkcja tworzaca array wspolrzednych
    def points_to_array(self, data):
        points = []
        for i in data:
            individual_point = []
            individual_point.append(i.x)
            individual_point.append(i.y)
            points.append(individual_point)
        return np.asarray(points)

    #glowny algorytm
    def train(self):
    # 1. Ustalamy liczbe skupien -> parametr konstruktora
    # 2. Ustalamy wstepne srodki skupien (centroidy) -> konstruktor
    # 3. 4. 5. powtarzamy w zaleznosci od liczby epok
        
        
        #plotujemy pierwszy wykres
        self.first_plot()

        while self.is_not_stabilized and (self.epoch_counter < self.number_of_epoch or self.epoch_limit == False):
            
            self.is_not_stabilized = False
            self.quantization_error = 0
            #czyscimy poprzednio przypisane punkty
            for center in self.centers:
                center.clear_assigned_points()
            
            # 3. Obliczamy odleglosc obiektow od srodkow skupien
            for point in self.points:
                point.closest_center(self.centers)
                 # 4. przypisujemy punkty do najblizszego centrum
                for center in self.centers:
                    
                    previous_centres = center.points_assigned
                    if (point.calculate_distance(center) <= point.minimal_distance):
                        center.assign_points(point) 
                    is_equal = False
                    if not is_equal:
                        self.is_not_stabilized = True

            # 5. Ustalamy nowe pozycje centroidow
            for center in self.centers:
                # previous_x = center.x
                # previous_y = center.y
                # previous_centres = center.points_assigned
                center.calculate_new_position()
                #obliczamy blad kwantylizacji
                center.calculate_error()
                self.quantization_error += center.quantization_error
            self.quantization_error /= len(self.points)
            #tu do wykresu liczba epok oraz wartosc bledu
            self.plot_error[0].append(self.epoch_counter + 1)
            self.plot_error[1].append(self.quantization_error)
            #zwiekszamy counter (to do nazw plikow i while'a)
            self.epoch_counter += 1

            #plotujemy wykres po kazdym przejsciu
            self.animation_plot()
            
            
        
        #po przejsciu tworzymy gif
        self.create_gif()
        #tworzymy wykres bledu
        self.plot_error_graph()

    #funkcja rysujaca pierwszy wykres (osobno zeby stan przed byl na szaro)
    def first_plot(self):
        points_plot = self.points_to_array(self.points)
        plt.plot(points_plot[:,0], points_plot[:,1], "^", markersize=0.5, color="gray")
        centers_plot = self.points_to_array(self.centers)
        plt.plot(centers_plot[:,0], centers_plot[:,1], ".", markersize=9, color="black")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("epoka " + str(self.epoch_counter))
        plt.savefig("Zad2/war_na_4/" + str(self.epoch_counter) + ".png")
        plt.clf()
        self.images.append(imageio.imread("Zad2/war_na_4/" + str(self.epoch_counter) + ".png"))
        # os.remove("Zad2/war_na_4/" + str(self.epoch_counter) + ".png")

    #funkcja rysujaca kolejne wykresy
    def animation_plot(self):
        centers_to_plot = self.points_to_array(self.centers)
        for center in self.centers:
            points_to_plot = self.points_to_array(center.points_assigned)
            plt.plot(points_to_plot[:,0], points_to_plot[:,1], "^", markersize=0.5)
        plt.plot(centers_to_plot[:,0], centers_to_plot[:,1], ".", markersize=9, color="black")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("epoka " + str(self.epoch_counter))
        plt.savefig("Zad2/war_na_4/" + str(self.epoch_counter) + ".png")
        plt.clf()
        self.images.append(imageio.imread("Zad2/war_na_4/" + str(self.epoch_counter) + ".png"))
        # os.remove("Zad2/war_na_4/" + str(self.epoch_counter) + ".png")

    #funkcja tworzaca gifa
    def create_gif(self):
        imageio.mimsave("Zad2/war_na_4/" + "animacja.gif", self.images, 'GIF', duration=0.5)

    #funkcja plotuje wykres bledu
    def plot_error_graph(self):
        plt.plot(self.plot_error[0], self.plot_error[1])
        plt.xlabel('Epoka')
        plt.ylabel('Wartosc')
        plt.title("Wartosc bledu kwantyzacji \n ilosc skupien: " + str(self.number_of_centroids))
        plt.savefig("Zad2/war_na_4/" + "wykres.png")
        plt.clf()

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

#generator wspolrzednych centroidow
def generate_random_points_in_range(x_min, x_max, y_min, y_max):
    x = random.randrange(x_min * 10000000000000, x_max * 10000000000000) / 10000000000000
    y = random.randrange(y_min * 10000000000000, y_max * 10000000000000) / 10000000000000
    return x, y

def main():

    NUMBER_OF_CENTROIDS = 8
    NUMBER_OF_EPOCH = 12

    kMean = KMean(points_matrix=create_points("Zad2/war_na_3/Danetestowe.txt", is_comma=True), 
                    number_of_centroids=NUMBER_OF_CENTROIDS, number_of_epoch=NUMBER_OF_EPOCH)
    
    kMean.train()

if __name__ == '__main__':
    main()