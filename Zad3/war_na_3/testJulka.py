import numpy

list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]

combined_data = list(zip(list1, list2))
print(combined_data)
xd = numpy.asarray(combined_data)
print(xd)
print(xd[:, 1])