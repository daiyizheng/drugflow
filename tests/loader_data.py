
import sys, os
sys.path.insert(0, "./")
import torch
from torchdrug import models, core
from drugflow.data.dataset import CustomMoleculeDataset
from drugflow.data.molecule import CustomMolecule
from drugflow.tasks.property_prediction import PropertyPrediction

class Mydata(CustomMoleculeDataset):
    target_fields = ["CT_TOX"]
    def __init__(self, path, verbose=1, **kwargs) -> None:
        super().__init__()
        if not os.path.exists(path):
            raise ValueError("暂时没有数据!!!")
        
        self.path = path

        self.load_csv(path, 
                      smiles_field="smiles", 
                      target_fields=self.target_fields,
                      verbose=verbose,
                      **kwargs)

if __name__ == '__main__':
    file_path = "/DYZ/dyz1/github_code/torchdrug/examples/datasets/clintox.csv"
    dataset = Mydata(file_path, 
                     atom_feature=["default", "center_identification"], 
                     molecular_feature=CustomMolecule) # , molecular_feature=CustomMolecule
    print(dataset)
    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    
    model = models.GIN(input_dim=dataset.node_feature_dim,
                   hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
    ## 构建task模型
    t = PropertyPrediction(model=model, 
                           criterion="ce", 
                           num_class=2, 
                           metric=("accuracy",),
                           task="CT_TOX")
    ## 训练
    optimizer = torch.optim.Adam(t.parameters(), lr=1e-3)
    solver = core.Engine(t, train_set, valid_set, test_set, optimizer,
                        gpus=[0], batch_size=1024)
    solver.train(num_epoch=100)
    solver.evaluate("valid")
