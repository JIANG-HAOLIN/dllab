import torch
from tqdm import tqdm
from models.architectures import *
from input_pipeline.datasets import *
from torch.nn import BCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluation.metrics import compute_matrix


import config


import cv2
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from input_pipeline.datasets import *
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit



opt = config.read_arguments()
opt.regression = True
device = opt.device
reg = opt.regression
root_path = opt.root_path
print(torch.cuda.is_available())

mdl = get_model(reg).to(device)

mdl.load_state_dict(
                                torch.load(root_path+'/'+opt.out_name + '.pth',
                                map_location=torch.device('cpu'))
                                 )



rgb_img = cv2.imread('./IDRID_dataset_preprocessed_test_size728x728/IDRiD_007.jpg', 1)[:, :, ::-1]
# rgb_img = cv2.resize(rgb_img, (512, 512))
rgb_img = np.float32(rgb_img) / 255
print(rgb_img.shape)

input_tensor = preprocess_image(rgb_img,
                                mean=[0.5424, 0.2638, 0.0875],
                                std=[0.4982, 0.4407, 0.2826],
                                )
# input_tensor = input_tensor.reshape(512,512,3)
print(input_tensor.shape)


device = "cpu"

trained_model_path = './grad-cam-try.pth.pth'

mdl = get_model(reg).to(device)

mdl.load_state_dict(torch.load(root_path+'/'+opt.out_name + '.pth',map_location=torch.device('cpu')))


mdl.eval()


target_layers = [mdl.features[-1]]
input_tensor = input_tensor# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=mdl, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# targets = [ClassifierOutputTarget(281)]
targets = None
cam.batch_size = 1
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.




grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()
print('done')

