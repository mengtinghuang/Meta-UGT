# Code for "In Silico Prediction of UGT-Mediated Metabolism in Drug-like Molecules via Graph Neural Network" this paper  
## Requipments  
* python 3.7
* pyTorch 1.5.0+
* DGL 0.5.2+
* dgllife 
* scikit learn
* rdkit
* numpy  
## Model for substrate/nonsubstrate classification  
We combined traditional machine learning methods and GNN methods to predict if a molecule is the substrate of UGT enzymes.  

We can trained the traiditional ML model as follow:
```bash
python find_best_model.py
```
and we can trained the GNN model as follow:
```bash
python best_GNN_model.py 
```


## Model for SOM prediction  
The GNN model automatically extracts atom environment features through convolutional layers. All the data has been processed, and We can train the model as follow:  
```bash
python SOM_train.py --train-path data/SOM/train.txt --val-path data/SOM/val.txt
```

Evalution:
```bash
python SOM_eval.py --test-path data/SOM/test.txt
```

## Trained model 
For substrate prediction model, all the trained models were saved on the file "model", and we can predict if a molecule is the substrate of UGT enzymes by applying the "pkl" file. For example:

```py
#python
import pickle
model = pickle.load(inputfile)
inputfile.close()
y_pred = model.predict(X)
```
For SOM model, We provide a jupyter notebook for predicting SOM metabolismed by UGT enzymes through our pre-trained models.
```bash
UGT_SOM.ipython
```
