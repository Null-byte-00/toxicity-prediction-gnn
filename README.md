# Machine learning model for drug toxicity prediction using Graph-based neural networks
this is a small graph-based model predicting substance toxicity
## The Model
the Model is make up of 2 Graph convolutional layers and two linear layers. it works by turning a molecule's chemical representation into a 1-hot encoded matrix (whuch represents atoms) and an edge matrix (which represents bonds)
## Dataset
I used [tox21](https://github.com/deepchem/deepchem/blob/master/datasets/tox21.csv.gz) dataset from [deepchem](https://github.com/deepchem/deepchem/) to train the model
## Training the model
for training the model you can you can run train.py or download the pre-trained model from the [release](https://github.com/Null-byte-00/toxicity-prediction-gnn/releases/latest/)
## Running the model
run test_model_cli.py
```
python test_model_cli.py
```
now enter SMILES representation of the substance:
```
Enter a SMILES string: CCN(CC)CCOC(=O)C1(C2CCCCC2)CCCCC1
Output: 52.33239531517029% chance of stress response to ATAD5.
```
