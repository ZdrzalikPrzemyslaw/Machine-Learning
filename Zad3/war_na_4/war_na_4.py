import numpy as np

def get_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) **2
    return np.sqrt(sum)

def kmeans(data, number_of_centers, max_ite):
    centroids = data[np.random.choice(range(len(data)), number_of_centers, replace=False)]

    is_Stabilized = False

    current_ite = 0

    while (not is_Stabilized) and (current_ite < max_ite):

        cluster_list = [[] for i in range(len(centroids))]

        for point in data:
            distance_list = []

            for center in centroids:
                distance_list.append(get_distance(center, point))

            cluster_list[int(np.argmin(distance_list))].append(point)

        cluster_list = list((filter(None, cluster_list)))

        previous_centroids = centroids.copy()

        centroids = []

        for i in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[i], axis=0))

        stop = np.abs(np.sum(previous_centroids) - np.sum(centroids))

        is_Stabilized = (stop < 0.001)

        current_ite += 1

    return np.array(centroids)

def read_2d_float_array_from_file(file_name):
    two_dim_list_of_return_values = []
    with open(file_name, "r") as file:
        lines = file.read().splitlines()
    for i in lines:
        one_dim_list = []
        for j in list(map(float, i.split())):
            one_dim_list.append(j)
        two_dim_list_of_return_values.append(one_dim_list)
    return np.asarray(two_dim_list_of_return_values)
def main():

    train_file = "Zad3/war_na_4/classification_train.txt"
    test_file = "Zad3/war_na_4/classification_test.txt"
    
    test = read_2d_float_array_from_file(train_file)
    test = kmeans(test[:,0:4],10,200)

    
    
    print("hello")

if __name__ == '__main__':
    main()
