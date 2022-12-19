import torch

def preprocess_input(input,label,device):
    # data['label'] = data['label'].long()
    # if opt.gpu_ids != "-1":
    #     data['label'] = data['label'].cuda()
    #     data['image'] = data['image'].cuda()
    # label_map = data['label']
    # if opt.gpu_ids != "-1":
    #     input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    # else:
    #     input_label = torch.LongTensor(batch_size, nc, h, w).zero_()
    # input_semantics = input_label.scatter_(1, label_map, 1.0)
    ####!!原本的label是一张图像!!通过scatter_函数转换成one-shot coding!!!
    input = input.to(device)
    batch_size, sequence_length, feature_channels = input.shape
    label = label.long()
    # label = label.view(batch_size,1).long()
    # ont_hot_init = torch.LongTensor(batch_size,feature_channels).zero_()
    # label = ont_hot_init.scatter_(1,label,1.0)
    label = label.to(device)
    return input, label