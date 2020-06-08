#### Usage
* Initializing the Network
```python
neurons = 5
siec = NeuralNetwork(number_of_neurons_hidden_layer=neurons, is_bias=True,
                     number_of_neurons_output=3, number_of_inputs=1)
train_file = "classification_train.txt"
num_of_iterations = 100
```
* Training the network

Use numpy delete to select data from input file
```python
siec.train(numpy.delete(read_2d_float_array_from_file(train_file), [0, 1, 2], 1)[:, :-1],
           read_2d_float_array_from_file(train_file)[:, -1:], num_of_iterations)
```
* Plotting the error
```python
plot_file()
```
* Plotting the change in correct assigments
```python
plot_file("correct_assigment.txt")
``` 