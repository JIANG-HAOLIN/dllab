from diabetic_retinopathy.models.architectures import get_model
import os
from diabetic_retinopathy.input_pipeline.datasets import *
from tqdm import tqdm
from metrics import compute_matrix


def ensemble_testing(path_checkpoints, device):

    model1 = get_model(pretrained=False, name="efficient_b3").to(device)
    model2 = get_model(pretrained=False, name="efficient_b3").to(device)
    model3 = get_model(pretrained=False, name="efficient_b3").to(device)
    model4 = get_model(pretrained=False, name="efficient_b3").to(device)
    model5 = get_model(pretrained=False, name="efficient_b3").to(device)

    path_model1 = os.path.join(path_checkpoints, "model1")
    path_model2 = os.path.join(path_checkpoints, "model2")
    path_model3 = os.path.join(path_checkpoints, "model3")
    path_model4 = os.path.join(path_checkpoints, "model4")
    path_model5 = os.path.join(path_checkpoints, "model5")

    model1_para = torch.load(path_model1)
    model1.load_state_dict(model1_para)
    model2_para = torch.load(path_model2)
    model2.load_state_dict(model2_para)
    model3_para = torch.load(path_model3)
    model3.load_state_dict(model3_para)
    model4_para = torch.load(path_model4)
    model4.load_state_dict(model4_para)
    model5_para = torch.load(path_model5)
    model5.load_state_dict(model5_para)

    mdls = [model1, model2, model3, model4, model5]

    voting = {"0": 0, "1": 0}
    test_accuracy = 0.

    for idx2, j in tqdm(enumerate(test_loader_ensemble)):
        with torch.no_grad():
            img = j[0]
            label = j[1]
            img = img.to(device)
            for mdl in mdls:
                mdl.eval()
                y = mdl(img)
                if y[0] <= 0.5:
                    voting["0"] += 1
                else:
                    voting["1"] += 1
            y = torch.tensor(0) if voting["0"] >= voting["1"] else 1

            label = label.to(torch.float).to(device)
            if y == label:
                test_accuracy += 1.
    result_accuracy = test_accuracy/len(test_loader_ensemble)
    print("ensemble learning accuracy:", result_accuracy)


ensemble_testing("path_to_checkpoints", "cuda")
