import random
import matplotlib.pyplot as pplt
import numpy


class Neuron():
    def __init__(self, x, w, d, step):
        self.x = x
        self.w = w
        self.u = 0
        self.d = d
        self.step = step
        self.y = 0

    def fun_u(self):
        self.u = 0
        for i in range(3):
            self.u = self.u+(self.w[i]*self.x[i])

    def fun_y(self):
        if self.u >= 0:
            self.y = 1
        else:
            self.y = 0

    def compare(self):
        if self.y == 0 and self.d == 1:
            for i in range(3):
                self.w[i] = self.w[i]+(self.step*self.x[i])
        elif self.y == 1 and self.d == 0:
            for i in range(3):
                self.w[i] = self.w[i]-(self.step*self.x[i])

    def calculate_points(self,wsp):
        return ((wsp*(-self.w[1]))-self.w[0])/self.w[2]

    def draw(self,i):
        pplt.plot([-5, 5], [-12, 8], "y--", label = "y = 2x-2")
        pplt.plot([-5,5],[self.calculate_points(-5), self.calculate_points(5)],label ="Stan Neuronu ")
        pplt.axis([-5,5,-5,5])
        pplt.title('Stan neuronu po '+str(i)+' powrórzeniach')
        pplt.legend()
        pplt.savefig('wykres'+str(i)+'.png')
        pplt.show()


w = []
x = []
d = []
step = 0.6

for i in range(0, 3):
    w.append((10 * numpy.random.random(1) - 5)[0])


data_file = open("dane.txt", "r")
data = data_file.readlines()
data_file.close()
lines = []
for line in data:
    lines.append(line.split(" "))
for i in lines:
    x.append([1, float(i[0]), float(i[1])])
    d.append(float(i[2]))
# x - punkty, d - czy pod czy nad
for i in range(0,100):
    if d[i] == 0:
        pplt.plot([x[i][1]], [x[i][2]], "bo")
    else:
        pplt.plot([x[i][1]], [x[i][2]], "ro")


neuron = Neuron(x[0], w, d[0], step)
print(neuron.w)
for i in range(0, 20000):
    neuron.x = x[i % 100]
    neuron.d = d[i % 100]
    # if i % 50 == 0:
    #     print(neuron.w)
    #     pplt.plot([-5, 5], [-12, 8], "y--")
    #     for j in range(0,100):
    #         if d[j] == 0:
    #             pplt.plot([x[j][1]], [x[j][2]], "bo")
    #         else:
    #             pplt.plot([x[j][1]], [x[j][2]], "ro")
    #     neuron.draw(i)
    neuron.fun_u()
    neuron.fun_y()
    neuron.compare()


pplt.plot([-5, 5], [-12, 8], "y--")
for i in range(0,100):
    if d[i] == 0:
        pplt.plot([x[i][1]], [x[i][2]], "bo")
    else:
        pplt.plot([x[i][1]], [x[i][2]], "ro")
neuron.draw(100)
print(neuron.w)


