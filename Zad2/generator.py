import random

from shapely import geometry

def choose_num_polygon_vertex():
    i = -1
    while i <= 2:
        print("Choose amount of polygon vertex\n"
              "Choose an Integer that is equal to or bigger than 3")
        try:
            i = int(input())
        except ValueError:
            pass
    return i


def choose_num_points():
    i = -1
    while i <= 0:
        print("Choose amount of points\n"
              "Choose an Integer that is equal to or bigger than 1")
        try:
            i = int(input())
        except ValueError:
            pass
    return i


def choose_points():
    num_points = choose_num_polygon_vertex()
    points = []
    counter = 0
    print("Fill in points as 'X Y',\nwhere X - x coordinate of point,\nY - y coordinate of point")
    while len(points) < num_points:
        print("Choose point nr ", counter + 1, ", out of ", num_points)
        counter += 1
        try:
            i = str(input())
            float_list = [float(x) for x in i.split(' ')]
            print(float_list)
            if len(float_list) == 2:
                points.append(float_list)
            else:
                print("fail")
                counter -= 1
        except ValueError:
            print("fail")
            counter -= 1
    return points


def generate_polygon():
    points = choose_points()
    poly = geometry.Polygon(points)
    return poly


def generate_points():
    poly = generate_polygon()
    amount_points = choose_num_points()
    points = []
    for i in range(amount_points):
        points.append(get_random_point_in_polygon(poly))
    return points


def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = geometry.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


def write_points_to_file(file_name):
    points = generate_points()
    with open(file_name, 'w') as file:
        for i in points:
            file.write(str(i.x) + " " + str(i.y) + "\n")


def main():
    write_points_to_file("plik.txt")


if __name__ == '__main__':
    main()