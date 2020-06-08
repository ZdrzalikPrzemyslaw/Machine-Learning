#### Usage
* Initializing the Network

In order to switch between aproximation and classification use the is_approximation bool
For approximation best number of output neurons is 1, for classification 3 output neurons are best
```python
train_file = "classification_train.txt"
test_file = "classification_test.txt"
data_input = read_2d_float_array_from_file(train_file)[:, :-1]
data_expected_output = read_2d_float_array_from_file(train_file)[:, -1]

siec = NeuralNetwork(number_of_neurons_hidden_layer=neurons, number_of_neurons_output=3,
                     is_bias=True, input_data=data_input,
                     expected_outputs=data_expected_output, is_aproximation=False)
iterations = 100
```
* Training the network

```python
siec.train(iterations)
```
* Plotting the error
```python
plot_file()
```