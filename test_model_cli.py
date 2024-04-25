from model import BaseModel

model = BaseModel(num_outputs=1, labels=['SR-ADAT5'], lr=0.00001)
model.load()

while True:
    smiles = input("Enter a SMILES string: ")
    output = model.forward_smiles(smiles)
    print(f"Output: {output}")