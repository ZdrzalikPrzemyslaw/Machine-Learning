#### Usage
* Initializing the Network
```python
neurons = 7
Network = NeuralNetwork(number_of_neurons_hidden_layer=neurons,
                     number_of_neurons_output=1, number_of_inputs=1, is_bias=True)
train_file = "approximation_train_1.txt"
iterations = 1000
```
* Training the network
```python
Network.train(read_2d_float_array_from_file(train_file)[:, 0],
                  read_2d_float_array_from_file(train_file)[:, 1], iterations)
```
* Plotting the error
```python
plot_file()
```
* Plotting the aproximated function
```python
# test_file = train_file
test_file = "approximation_test.txt"
plot_function(Network, train_file, neurons, test_file)
``` 