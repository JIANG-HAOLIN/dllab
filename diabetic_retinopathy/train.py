from models.architectures import MyModel
from input_pipeline.datasets import *

mdl = MyModel(3)

for i in train_loader:
    print(i.shape)
    y = mdl(i)
    print("output", y.shape)