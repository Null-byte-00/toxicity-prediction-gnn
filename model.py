import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch.optim import SGD
from process import smiles_to_onehot, smiles_to_edge


class BaseModel(nn.Module):
    def __init__(self, max_atoms=300, in_features=8,hidden_features=8, num_outputs=5, 
                 labels = [],lr=0.0000001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.num_outputs = num_outputs
        self.labels = labels
        self.max_atoms = max_atoms

        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.flatten = nn.Flatten( start_dim=0)
        middle_features = max_atoms * hidden_features
        print(middle_features)
        self.linear1 = nn.Linear(middle_features, middle_features)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(middle_features, num_outputs)

        self.optimizer = SGD(self.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss() 

    def forward(self, x, edge):
        x = self.conv1(x, edge).relu()
        x = self.conv2(x, edge).relu()
        x = self.flatten(x)
        x = self.linear1(x).relu()
        #x = self.dropout(x)
        x = self.linear2(x).sigmoid()
        return x
    
    def forward_smiles(self, smiles):
        onehot = smiles_to_onehot(smiles, max_atoms=self.max_atoms)
        edge = smiles_to_edge(smiles)
        output = self.forward(onehot, edge)
        return output
    
    def train_smiles(self, smiles, target, verbose=True):
        target = torch.tensor(target, dtype=torch.float32)
        onehot = smiles_to_onehot(smiles, max_atoms=self.max_atoms)
        edge = smiles_to_edge(smiles)
        output = self.forward(onehot, edge)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        if verbose:
            print(f"output: {output}")
            print(f"target: {target}")
            print(f"loss: {loss}")
        return loss

    
    def labels_to_target(self, labels):
        target_dict = {}
        for index, label in enumerate(self.labels):
            target_dict[label] = index
        out_tensor = torch.zeros(self.num_outputs)
        for target in target_dict:
            if target in labels:
                out_tensor[target_dict[target]] = 1
        return out_tensor

    def save(self, file_name='models/model.pth'):
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='models/model.pth'):
        self.load_state_dict(torch.load(file_name))

