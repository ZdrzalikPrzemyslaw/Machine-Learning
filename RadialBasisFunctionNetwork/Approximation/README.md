#### Usage
* Initializing the Network
```python
neurons = 8
train_file = "approximation_train_1.txt"
test_file = "approximation_test.txt"
Network = NeuralNetwork(neurons, 1, True, read_2d_float_array_from_file(train_file)[:, 0],
                     read_2d_float_array_from_file(train_file)[:, 1])
```
* Training the network
```python
Network.train(iterations)
```
* Plotting the error
```python
plot_file()
```
* Plotting the aproximated function
```python
plot_function(Network, train_file, neurons, test_file)
``` 