import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from diabetic_retinopathy.models.architectures import *
from diabetic_retinopathy import config

from efficientnet_pytorch import EfficientNet







def image_preprocessing(img,mean,std):
    img=np.float32(img)/255
    preprocessed_img=torch.flip(torch.Tensor(img.copy()),dims=[2])
    preprocessed_img=((preprocessed_img-mean)/std).permute(2,0,1)
    preprocessed_img=torch.unsqueeze(preprocessed_img,0)
    inputs = preprocessed_img.requires_grad_(True)
    return inputs,img


class grad_cam_model:
    def __init__(
            self, model,
                 block_name,
                 target_layer_names,
                img_size
                 ):
        self.img_size = img_size
        self.model=model
        self.target_layer_names =target_layer_names
        self.model.eval()
        self.gradss=get_grad(self.model, block_name,target_layer_names)

    def compute_grad_cam_each_layer(self):
        grad_cam_each_layer = {}
        for idx, feature in enumerate(self.features):
            num_layer = len(self.features)
            grads_val = self.gradss.gradients[num_layer - 1 - idx].cpu().data.numpy()##compute the gradients
            weights = np.mean(grads_val, axis=(2, 3))[0, :]##average of gradients
            target = self.features[idx]
            target = target.cpu().data.numpy()[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for idx_channel, weights_channel in enumerate(weights):## for each channel
                cam += weights_channel * target[idx_channel, :, :]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (self.img_size, self.img_size))
            cam = (cam - np.min(cam))
            cam = cam/np.max(cam)
            grad_cam_each_layer[self.target_layer_names[idx]] = cam
        return grad_cam_each_layer

    def __call__(
            self,
                 inputs,
                 index=None,
                 img=None
                 ):
        self.features, output =self.gradss(inputs)
        target_class= torch.Tensor([[1]]).requires_grad_(True)
        target_class =torch.sum(target_class * output)##get the one-hot coding for output
        self.model.zero_grad()
        target_class.backward()
        self.model.zero_grad()
        grad_cam_each_layer =self.compute_grad_cam_each_layer()
        for name, grad_cam in grad_cam_each_layer.items():
            raw_gradcam = np.float32(cv2.applyColorMap(np.uint8(255*grad_cam), cv2.COLORMAP_JET))/255
            grd_cm=(raw_gradcam+np.float32(img))
            grd_cm=grd_cm/np.max(grd_cm)
            cv2.imwrite("gram_cam{}.jpg".format(name), np.uint8(255 *grd_cm))
        return grad_cam_each_layer


class Guided_backpropagation_model:
    def __init__(self,
                 model,
                 activation_layer_name='SiLU'
                 ):
        self.model =model
        self.model.eval()
        override_act_func =Guided_backpropagation_relu.apply
        self.override_activation(override_act_func,activation_layer_name)

    def override_activation(self,override_act_func,activation_layer_name):
        for idx0, module0 in self.model._modules.items():
            module0 = self.model._modules[idx0]
            if module0.__class__.__name__ == activation_layer_name:
                self.model._modules[idx0] = override_act_func
            for idx1, _ in module0._modules.items():
                module1 = module0._modules[idx1]
                if module1.__class__.__name__ == activation_layer_name:
                    self.model._modules[idx0]._modules[idx1] = override_act_func
                    continue
                for idx2, _ in module1._modules.items():
                    module2 = module1._modules[idx2]
                    if module2.__class__.__name__ == activation_layer_name:
                        self.model._modules[idx0]._modules[idx1]._modules[idx2] = override_act_func
                    for idx3, _ in module2._modules.items():
                        module3 = module2._modules[idx3]
                        if module3.__class__.__name__ == activation_layer_name:
                            self.model._modules[idx0]._modules[idx1]._modules[idx2]._modules[idx3] = override_act_func
                        for idx4, _ in module3._modules.items():
                            module4 = module3._modules[idx4]
                            if module4.__class__.__name__ == activation_layer_name:
                                self.model._modules[idx0]._modules[idx1]._modules[idx2]._modules[idx3]._modules[idx4] = override_act_func
                            for idx5, _ in module4._modules.items():
                                module5 = module4._modules[idx5]
                                if module5.__class__.__name__ == activation_layer_name:
                                    self.model._modules[idx0]._modules[idx1]._modules[idx2]._modules[idx3]._modules[idx4]._modules[idx5] = override_act_func

    def __call__(self, inputs, index=None):
        output = self.model(inputs)
        target_class = torch.Tensor([[1]]).requires_grad_(True)
        target_class = torch.sum(target_class * output)
        target_class.backward()
        gradient = inputs.grad.cpu().data.numpy()[0, :, :, :]
        generate_guided_backpropagation(gradient)
        return gradient.transpose((1, 2, 0))

def generate_guided_grad_cam(idx,mask,guided_backpropagation_result):
    grad_cam_mask = cv2.merge([mask, mask, mask])
    product = grad_cam_mask* guided_backpropagation_result
    product = (product - np.mean(product))/ (np.std(product) + 1e-5)
    guided_grad_cam = np.clip((product * 0.1 + 0.5), 0, 1)
    guided_grad_cam = (guided_grad_cam * 255)
    cv2.imwrite('guided_backpropagation{}.jpg'.format(idx), guided_grad_cam)

def generate_guided_backpropagation(guided_backpropagation_result):
    input = guided_backpropagation_result
    input = (input - np.mean(input))/ (np.std(input) + 1e-5)
    result = np.clip((input * 0.1 + 0.5), 0, 1)
    guided_backpropagation_image =(result * 255)
    cv2.imwrite('guided_backpropagation.jpg', guided_backpropagation_image)


class get_grad():
    def __init__(self, model, block_name, target_layers):
        self.model = model
        self.block_name = block_name
        self.target_layers = target_layers
        self.gradients = []
    def save_gradient(self, grad):
        self.gradients.append(grad)
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for idx, module in self.model._modules.items():##idx='features' for efficientent
            if idx == self.block_name:
                for name, block in enumerate(getattr(self.model, self.block_name)):
                    #(name=0,block= each sequential in features block)
                    x = block(x)
                    if str(name) in self.target_layers:
                        x.register_hook(self.save_gradient)
                        outputs += [x]
            elif idx != self.block_name:
                try:
                    x = module(x)
                except:
                    x = x.view(x.size(0), -1)##makes _modules sequential
                    x = module(x)
        return outputs, x

class Guided_backpropagation_relu(Function):
    @staticmethod
    def forward(self, i):
        grad__output = i * (i > 0).type_as(i) + torch.zeros(i.size()).type_as(i)
        self.save_for_backward(i)
        return grad__output
    @staticmethod
    def backward(self, grad_output):
        i = self.saved_tensors[0]
        grad__input = grad_output*(i > 0).type_as(grad_output) + torch.zeros(i.size()).type_as(i)
        grad__input = grad__input*(grad_output > 0).type_as(grad_output) + torch.zeros(i.size()).type_as(i)
        return grad__input

if __name__ == '__main__':
    opt = config.read_arguments()
    device = opt.device
    root_path = opt.root_path
    wanted_size = 600
    batch_size = opt.batch_size

    pth = '/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15p(2)/diabetic_retinopathy/pth1all_600_b7batch4_93_880_.pth'
    # mdl = get_multi_model(3).to('cpu')
    mdl = get_model_b7().to('cpu')
    mdl.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    mdl.eval()
    img = cv2.imread('/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15p(2)/diabetic_retinopathy/IDRID_dataset_preprocessed_test_size600x600/IDRiD_102.jpg', 1)
    print(img.shape)
    mean=torch.Tensor([0.5424, 0.2638, 0.0875])
    std=torch.Tensor([0.4982, 0.4407, 0.2826])
    inputs,img = image_preprocessing(img,mean,std)
    target_index = 0

    grad_cam_mdl = grad_cam_model(model=mdl, block_name='features', target_layer_names=['1', '2','3','4','5','6','7','8','9','10'],img_size=600)
    grad_cam_outputs = grad_cam_mdl(inputs, target_index,img=img)
    guided_backpropagation_mdl = Guided_backpropagation_model(model=mdl, activation_layer_name='SiLU')
    guided_backpropagation_outputs = guided_backpropagation_mdl(inputs, index=target_index)
    for idx_layer, grad_cam_each_layer in grad_cam_outputs.items():
        generate_guided_grad_cam(idx_layer,grad_cam_each_layer,guided_backpropagation_outputs)
