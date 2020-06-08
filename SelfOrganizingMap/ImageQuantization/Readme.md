#### Usage
* Choose the picture to quantize

```python
im = Image.open('Images/3399232.jpg')
```

* Initializing the Network
```python
kohonen = KohonenOrNeuralGas(input_matrix=image_pixels_to_array(im),
                                 neuron_num=9,
                                 is_gauss=True, is_neural_gas=True, epoch_count=1, neighbourhood_radius=1.5,
                                 min_potential=0, alfa=0.8)
```

* Train the network
```python
kohonen.train()
```

* Save Picture 
```python
save_new_picture(im, kohonen)
```