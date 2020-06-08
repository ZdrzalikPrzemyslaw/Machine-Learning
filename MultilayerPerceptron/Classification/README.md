#### Usage
* Initializing the Network
```python
neurons = 3
Network = NeuralNetwork(number_of_neurons_hidden_layer=neurons, is_bias=True,
                        number_of_neurons_output=4, number_of_inputs=4)
input_data = "dane.txt"
iterations = 1000
```
* Training the network

```python
Network.train(read_2d_int_array_from_file(input_data), read_2d_int_array_from_file(input_data).T, iterations,
                  ERROR_FILE)
```
* Plotting the error
```python
plot_file(ERROR_FILE)
```
* Visualizing the transformation
```python
print("Input Data:")
inpuciki = numpy.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print(inpuciki)
print("Output Data:")
print(Network.calculate_outputs(inpuciki)[1])
``` 