from models.architectures import MyModel
from input_pipeline.datasets import *

device = "cuda"

mdl = MyModel(3).to(device)

for i in train_loader:
    img = i[0]
    label = i[1]
    img = img.to(device)
    y = mdl(img)
    print(y.shape)
    break
