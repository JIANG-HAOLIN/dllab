import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from evaluation.confusion_matrix import generate_confusion_Matrix


def visualization(usr='25',exp='50',result_dict = None,true_dict=None):
    accx_list = []
    accy_list = []
    accz_list = []
    gyrox_list = []
    gyroy_list = []
    gyroz_list = []
    result_dict = np.load(result_dict,allow_pickle=True).item()
    true_dict = np.load(true_dict,allow_pickle=True).item()
    label_dict = {0:'walking',1:'walking_upstairs',2:'walking_downstairs',3:'sitting',4:'standing',5:'laying',
            6:'stand_to_sit',7:'sit_to_stand',8:'sit_to_lie',9:'lie_to_sit',10:'stand_to_lie',11:'lie_to_stand'}
    color_dict={0:'red',1:'orange',2:'yellow',3:'greenyellow',4:'springgreen',5:'aquamarine',
                6:'cyan',7:'skyblue',8:'mediumpurple',9:'violet',10:'magenta',11:'hotpink'}
    # print(result_dict[usr][exp])

    for user,exps in result_dict.items():
        exp = exps
        acc_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/acc_exp{exp}_user{usr}.txt'
        gyrp_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/gyro_exp{exp}_user{usr}.txt'
        with open(acc_path) as acc:
            file_acc = acc.readlines()
        with open(gyrp_path) as gyro:
            file_gyro = gyro.readlines()

        for line_index in range(len(file_acc)):
            # float_list.append(list(map(float, (file_acc[line_index].split()))) + list(map(float, (file_gyro[line_index].split()))))
            accx_list.append(list(map(float, (file_acc[line_index].split())))[0])
            accy_list.append(list(map(float, (file_acc[line_index].split())))[1])
            accz_list.append(list(map(float, (file_acc[line_index].split())))[2])

            gyrox_list.append(list(map(float, (file_gyro[line_index].split())))[0])
            gyroy_list.append(list(map(float, (file_gyro[line_index].split())))[1])
            gyroz_list.append(list(map(float, (file_gyro[line_index].split())))[2])


    # print(acc_list[0])
    fig, ax = plt.subplots(3,1,figsize=(16,9), gridspec_kw={'height_ratios': [4,4,1]})

    ax[0].set_title('acceleration',fontdict={'fontsize': 20,'fontweight' : 40})
    ax[0].plot(range(len(accx_list)), accx_list, label="acc_x")  # plot
    ax[0].plot(range(len(accy_list)), accy_list, label="acc_y")  # plot
    ax[0].plot(range(len(accz_list)), accz_list, label="acc_z")  # plot

    ax[1].set_title('gyroscope',fontdict={'fontsize': 20,'fontweight' : 40})
    ax[1].plot(range(len(gyrox_list)), gyrox_list, label="gyro_x")  # plot
    ax[1].plot(range(len(gyroy_list)), gyroy_list, label="gyro_y")  # plot
    ax[1].plot(range(len(gyroz_list)), gyroz_list, label="gyro_z")  # plot


    cmap = matplotlib.colors.ListedColormap(['red','orange','yellow','greenyellow','springgreen','aquamarine',
                'cyan','skyblue','mediumpurple','violet','magenta','hotpink'])
    bounds = [0, 1, 2, 3, 4,5,6,7,8,9,10,11]
    # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    result_label = []
    true_label =[]
    for data in result_dict[usr][exp]:
        start = data[0][0]
        end = data[0][1]
        color = color_dict[int(data[1])]
        ax[0].axvspan(start, end , alpha=0.3, color=color,lw=0)
        ax[0].legend(loc=1)
        ax[1].axvspan(start, end , alpha=0.3, color=color,lw=0)
        ax[1].legend(loc=1)
        result_label.append(int(data[1]))

    for data in true_dict[usr][exp]:
        true_label.append(int(data[1]))


    print(result_label,true_label)

    interval = len(accx_list)/12
    ax[2].set_ylim(0,1)
    ax[2].set_xlim(0,len(accx_list))
    # ax[2].axis('off')
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    ls=[]
    for i,color in color_dict.items():
        ax[2].axvspan(i*interval, (i+1)*interval, alpha=0.3, color=color,lw=0)
        ls.append((i*interval+(i+1)*interval)/2,)


## confusion Matrix for single file
    ax[2].set_xticks(ls,['walking','walking_upstairs','walking_downstairs','sitting','standing','laying',
            'stand_to_sit','sit_to_stand','sit_to_lie','lie_to_sit','stand_to_lie','lie_to_stand'],rotation=15)
    plt.subplots_adjust(hspace=0.4)
    fig.show()
    result_label = np.array(result_label)
    true_label = np.array(true_label)
    label_list = []
    all_possible_labels = np.concatenate((result_label,true_label))
    for i in range(12):
        if i in all_possible_labels:
                label_list.append(label_dict[i])
    # print(label_list)
    generate_confusion_Matrix(true_label, result_label,label_list=label_list)

## confusion Matrix for whole test set
    results = []
    y_true = []
    for usr,files in result_dict.items():
        for file,datas in files.items():
            for data in datas:
                results.append(int(data[1]))
    for usr,files in true_dict.items():
        for file,datas in files.items():
            for data in datas:
                y_true.append(int(data[1]))
    results = np.array(results)
    y_true = np.array(y_true)
    label_list = []
    all_possible_labels = np.concatenate((results,y_true))
    for i in range(12):
            if i in all_possible_labels:
                    label_list.append(label_dict[i])
    generate_confusion_Matrix(y_true, results,label_list=label_list)

def show_confusion_matrix(result_dict,true_dict,how):
    label_dict = {0:'walking',1:'walking_upstairs',2:'walking_downstairs',3:'sitting',4:'standing',5:'laying',
            6:'stand_to_sit',7:'sit_to_stand',8:'sit_to_lie',9:'lie_to_sit',10:'stand_to_lie',11:'lie_to_stand'}
    result_dict = np.load(result_dict,allow_pickle=True).item()
    true_dict = np.load(true_dict,allow_pickle=True).item()
    result_label = []
    true_label =[]
    ## confusion Matrix for whole test set
    results = []
    y_true = []
    for usr, files in result_dict.items():
        for file, datas in files.items():
            for data in datas:
                results.append(int(data[1]))
    for usr, files in true_dict.items():
        for file, datas in files.items():
            for data in datas:
                y_true.append(int(data[1]))
    results = np.array(results)
    y_true = np.array(y_true)
    label_list = []
    all_possible_labels = np.concatenate((results, y_true))
    for i in range(12):
        if i in all_possible_labels:
            label_list.append(label_dict[i])
    generate_confusion_Matrix(y_true, results, label_list=label_list)


visualization(result_dict='result_dict_s2l.npy',true_dict='true_dict_s2l.npy')