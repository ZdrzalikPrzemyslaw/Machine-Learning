import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance

class KMean:
    def __init__(self, input_matrix, number_of_centroids):
        #input matrix [x, y, centroid]
        self.input_matrix = input_matrix
        #liczba centroidow
        self.number_of_centroids = number_of_centroids
        #losujemy startowe pozycje centroidow
        self.centroid = np.random.normal(np.mean(input_matrix), np.std(input_matrix), 
                                        size=(self.number_of_centroids, len(input_matrix[0])-1))
        #odleglosc punktow od poszczegolnych centroidow
        self.distance_to_centr = np.zeros(len(input_matrix))
    
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
    
    #okreslamy przynaleznosc punktow do danych centroidow na podstawie odleglosci
    def points_to_centr(self):
        j=0
        for i in self.distance_to_centr:
            for k in range (len(i)):
                if (i.min() == i[k]):
                    self.input_matrix[j,2] = k+1
            j+=1
        
        

        
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


def main():
    NUMBER_OF_CENTROIDS = 4
    kMean = KMean(input_matrix=read_2d_float_array_from_file("Zad2\war_na_3\Danetestowe.txt", is_comma=True),
                number_of_centroids=NUMBER_OF_CENTROIDS)
    kMean.calculate_distance()
    kMean.points_to_centr()
    print("Hello")

if __name__ == '__main__':
    main()