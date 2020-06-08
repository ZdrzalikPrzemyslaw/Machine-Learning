#### Usage
* Run generator.py to generate a custom figure

* Initializing the Network
```python
NUMBER_OF_CENTROIDS = 12
NUMBER_OF_EPOCH = 14

kMean = KMean(points_matrix=create_points("Danetestowe.txt", is_comma=True),
              number_of_centroids=NUMBER_OF_CENTROIDS, number_of_epoch=NUMBER_OF_EPOCH)
```

* Train the network and generate animation
```python
kMean.train()
```
