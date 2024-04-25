from model import BaseModel
import pandas as pd
from matplotlib import pyplot as plt
import torch


ITERATIONS = 2


def oversample(dataframe):
    """
    Oversample the positive class to balance the dataset.
    """
    positive = dataframe[dataframe["SR-ATAD5"] == 1]
    negative = dataframe[dataframe["SR-ATAD5"] == 0]
    ratio = len(negative) / len(positive)
    print(f"Ratio: {ratio}")
    oversampled = pd.concat([positive] * int(ratio), ignore_index=True)
    return pd.concat([oversampled, negative], ignore_index=True)


dataframe = pd.read_csv('datasets/tox21.csv')
dataframe = oversample(dataframe)
dataframe = dataframe.sample(frac=1)

train, test = dataframe[:int(len(dataframe) * 0.8)], dataframe[int(len(dataframe) * 0.8):]

smiles_train, smiles_test = train['smiles'], test['smiles']
sratad5_train, sratad5_test = train["SR-ATAD5"], test["SR-ATAD5"]

model = BaseModel(num_outputs=1, labels=['SR-ADAT5'], lr=0.00001)

losses = []

for _ in range(ITERATIONS):
    for i in train.index:
        if sratad5_train[i] == sratad5_train[i]:
            print(f"Training on {smiles_train[i]} with {sratad5_train[i]}.")
            loss = model.train_smiles(smiles_train[i], [sratad5_train[i]])


for i in test.index:
    if sratad5_test[i] == sratad5_test[i]:
        output = model.forward_smiles(smiles_test[i])
        expected = sratad5_test[i]
        loss = (output - expected) ** 2
        losses.append(loss.item())


avg_loss = sum(losses) / len(losses)
print(f"Average loss: {avg_loss}")

plt.plot(losses)
plt.show()

model.save()