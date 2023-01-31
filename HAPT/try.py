# for i in range(2):
#     print(1)
#
# string_list = ['1234 12354 12355','4567','34656']
# print(list(map(float,string_list[0].split())))



# result_dict = {
#                 f'{s}': {f'{e}':[] for e in (2*s,2*s+1)} for s in range(22,28)
#                }
# print(result_dict)
#{'22': {'44': [], '45': []}, '23': {'46': [], '47': []}, '24': {'48': [], '49': []}, '25': {'50': [], '51': []}, '26': {'52': [], '53': []}, '27': {'54': [], '55': []}}


# import torch
# a = torch.randn(1,2,3)
# print(int(a[0][0][0]))


import matplotlib
import matplotlib.pyplot as plt
def visualization(usr='22',exp='44',target='acc',result=None):
    acc_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/acc_exp{exp}_user{usr}.txt'
    gyrp_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/gyro_exp{exp}_user{usr}.txt'
    pred_result = result
    with open(acc_path) as acc:
        file_acc = acc.readlines()
    with open(gyrp_path) as gyro:
        file_gyro = gyro.readlines()
    acc_list = []
    gyro_list = []
    for line_index in range(len(file_acc)):
        acc_list.append(list(map(float, (file_acc[line_index].split()))))
        gyro_list.append(list(map(float, (file_gyro[line_index].split()))))

    l = eval(target+'_list')
    g = gyro_list


    color_dict={0:''}

    fig1 = plt.figure()
    plt.plot(range(len(l)), l, label=target)
    fig1.show()

    fig2 = plt.figure()
    plt.plot(range(len(g)), g, label='gyro')
    fig2.show()

visualization(result=[[300,12000],3])