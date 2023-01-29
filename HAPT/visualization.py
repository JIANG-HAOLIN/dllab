import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def visualization(usr='22',exp='44',result_dict = None):
    result_dict = np.load(result_dict,allow_pickle=True).item()
    color_dict={0:'red',1:'orange',2:'yellow',3:'greenyellow',4:'springgreen',5:'aquamarine',
                6:'cyan',7:'skyblue',8:'mediumpurple',9:'violet',10:'magenta',11:'hotpink'}
    # print(result_dict[usr][exp])
    acc_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/acc_exp{exp}_user{usr}.txt'
    gyrp_path = f'/Users/hlj/Documents/NoSync.nosync/DL_Lab/dl-lab-22w-team15/HAPT/RawData/gyro_exp{exp}_user{usr}.txt'
    with open(acc_path) as acc:
        file_acc = acc.readlines()
    with open(gyrp_path) as gyro:
        file_gyro = gyro.readlines()
    accx_list = []
    accy_list = []
    accz_list = []
    gyrox_list = []
    gyroy_list = []
    gyroz_list = []
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

    for data in result_dict[usr][exp]:
        start = data[0][0]
        end = data[0][1]
        color = color_dict[int(data[1])]
        ax[0].axvspan(start, end , alpha=0.3, color=color,lw=0)
        ax[0].legend(loc=1)
        ax[1].axvspan(start, end , alpha=0.3, color=color,lw=0)
        ax[1].legend(loc=1)


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

    ax[2].set_xticks(ls,['walking','walking_upstairs','walking_downstairs','sitting','standing','laying',
            'stand_to_sit','sit_to_stand','sit_to_lie','lie_to_sit','stand_to_lie','lie_to_stand'],rotation=15)
    plt.subplots_adjust(hspace=0.4)
    fig.show()

visualization(result_dict='result_dict.npy')