#### Usage
* Initializing the Network
```python
train_file = "classification_train.txt"
test_file = "classification_test.txt"
data_input = read_2d_float_array_from_file(train_file)[:, :-1]
data_expected_output = read_2d_float_array_from_file(train_file)[:, -1]
siec = NeuralNetwork(number_of_neurons_hidden_layer=10, number_of_neurons_output=3,
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
* Generating confusion matrix
```python
if not siec.is_aproximation:
    siec.plot_classification()
    correct_amount = 0
    all_1 = [0, 0, 0]
    all_2 = [0, 0, 0]
    all_3 = [0, 0, 0]
    it = 0
    if len(siec.output_layer) == 3:
        for i in read_2d_float_array_from_file(test_file)[:, :]:
            obliczone = siec.calculate_outputs(i[0:4])[1]
            if i[-1] == 1:
                all_1[numpy.argmax(obliczone)] += 1
            elif i[-1] == 2:
                all_2[numpy.argmax(obliczone)] += 1
            elif i[-1] == 3:
                all_3[numpy.argmax(obliczone)] += 1
            it += 1
    else:
        for i in read_2d_float_array_from_file(test_file)[:, :]:
            obliczone = siec.calculate_outputs(i[0:4])[1]
            classa = 0
            if abs(obliczone - 1) <= 0.5:
                classa = 1
            elif abs(obliczone - 2) <= 0.5:
                classa = 2
            elif abs(obliczone - 3) <= 0.5:
                classa = 3

            if i[-1] == classa:
                correct_amount += 1

            if i[-1] == 1:
                all_1[classa - 1] += 1
            elif i[-1] == 2:
                all_2[classa - 1] += 1
            elif i[-1] == 3:
                all_3[classa - 1] += 1
            it += 1
    print("Classification Objects  :   1,  2,  3")
    print("Classification Object 1 : ", all_1)
    print("Classification Object 2 : ", all_2)
    print("Classification Object 3 : ", all_3)
``` 