# [MLP](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/tree/master/MultilayerPerceptron)

##[Approximation](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/tree/master/MultilayerPerceptron/Approximation)

#### Example output

![Plot](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/blob/master/.github/Approximation_MLP_Example_Plot.png)
![Error](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/blob/master/.github/Approximation_MLP_Example_Error.png)


#### Usage
* Initializing the Network
```python
neurons = 7
siec = NeuralNetwork(number_of_neurons_hidden_layer=neurons, 
                     number_of_neurons_output=1, number_of_inputs=1, is_bias=True)
train_file = "train_points.txt"
iterations = 1000
```
* Training the network
```python
siec.train(read_2d_float_array_from_file(train_file)[:, 0], read_2d_float_array_from_file(train_file)[:, 1],
               iterations)
```
* Plotting the error
```python
plot_file()
```
* Plotting the aproximated function
```python
# test_file = train_file
test_file = "approximation_test.txt"
plot_function(siec, train_file, neurons, test_file)
``` 


* [Classification](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/tree/master/MultilayerPerceptron/Classification)
* [Transformation](https://github.com/ZdrzalikPrzemyslaw/Machine-Learning/tree/master/MultilayerPerceptron/Transformation)