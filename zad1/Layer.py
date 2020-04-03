import numpy

class Neuron:
    # TROCHĘ TUTAJ OSZUKUJE Z TYM __repr__
    def __repr__(self):
        return str(self.weights)

    def __str__(self):
        return str(self.weights)

    def __init__(self, num_of_weights, if_bias):
        self.weights = 2 * numpy.random.random(num_of_weights) - 1
        self.bias = 0
        if if_bias:
            self.bias =  2 * numpy.random.random(1) - 1


class Layer:
    def __repr__(self):
        return "Instance of Layer"

    def __str__(self):
        return str(self.neurons)

    def __init__(self, num_of_weights, if_bias, num_of_neurons):
        neurons = []
        for i in range(0, num_of_neurons):
            neurons.append(Neuron(num_of_weights, if_bias))
        self.neurons = numpy.asarray(neurons)


def __main():
    print(Layer(2, True, 2))

if __name__ == "__main__":
    __main()