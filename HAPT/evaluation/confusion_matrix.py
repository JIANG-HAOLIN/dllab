import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import DataFrame
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix
import seaborn
import string
from string import ascii_uppercase


def generate_confusion_Matrix(
    y_test,
    predictions,
    label_list = None
):
    color_map="Greens"
    columns = ["class %s" % (i) for i in list(ascii_uppercase)[0 : len(np.unique(label_list))]]
    confm = confusion_matrix(y_test, predictions)
    dataframe = DataFrame(confm, index=columns, columns=columns)

    font_size=8
    dataframe = dataframe.T
    sum_col = []
    for c in dataframe.columns:
        sum_col.append(dataframe[c].sum())
    sum_lin = []
    for item_line in dataframe.iterrows():
        sum_lin.append(item_line[1].sum())
    dataframe["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    dataframe.loc["sum_col"] = sum_col


    fig = plt.figure("Conf matrix default", [8,8])
    ax1 = fig.gca()
    ax1.cla()

    ax = seaborn.heatmap(
        dataframe,
        annot=True,
        annot_kws={"size": font_size},
        linewidths=0.5,
        ax=ax1,
        cbar=False,
        cmap=color_map,
        linecolor="w",
        fmt=".2f",
    )


    ax.set_xticks(np.linspace(0.5,len(label_list)-0.5,len(label_list)), label_list,
                     rotation=45)
    ax.set_yticks(np.linspace(0.5,len(label_list)-0.5,len(label_list)), label_list,
                     rotation=25)

    for axis_ticks in [ax.xaxis.get_major_ticks(),ax.yaxis.get_major_ticks()]:
        for tick in axis_ticks:
            tick.tick1On = False
            tick.tick2On = False


    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    data_seq = np.array(dataframe.to_records(index=False).tolist())
    more = []
    less = []
    position = -1
    for out_text in ax.collections[0].axes.texts:
        pos = np.array(out_text.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        position += 1
        more1 = []
        less1 = []
        cell_value = data_seq[lin][col]
        tot_all = data_seq[-1][-1]
        percentage = (float(cell_value) / tot_all) * 100
        curr_column = data_seq[:, col]
        curr_column_length = len(curr_column)

        if (col == (curr_column_length - 1)) or (lin == (curr_column_length - 1)):
            if cell_value != 0:
                if (col == curr_column_length - 1) and (lin == curr_column_length - 1):
                    tot_rig = 0
                    for i in range(data_seq.shape[0] - 1):
                        tot_rig += data_seq[i][i]
                    percentage_ok = (float(tot_rig) / cell_value) * 100
                elif col == curr_column_length - 1:
                    tot_rig = data_seq[lin][lin]
                    percentage_ok = (float(tot_rig) / cell_value) * 100
                elif lin == curr_column_length - 1:
                    tot_rig = data_seq[col][col]
                    percentage_ok = (float(tot_rig) / cell_value) * 100
                percentage_err = 100 - percentage_ok
            else:
                percentage_ok = percentage_err = 0

            percentage_ok_s = ["%.2f%%" % (percentage_ok), "100%"][percentage_ok == 100]

            less1.append(out_text)

            font_prop = font_manager.FontProperties(weight="bold", size=font_size)
            text_kwargs = dict(
                color="w",
                ha="center",
                va="center",
                gid="sum",
                fontproperties=font_prop,
            )
            lis_txt = ["%d" % (cell_value), percentage_ok_s, "%.2f%%" % (percentage_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy()
            dic["color"] = "g"
            lis_kwa.append(dic)
            dic = text_kwargs.copy()
            dic["color"] = "r"
            lis_kwa.append(dic)
            lis_pos = [
                (out_text._x, out_text._y - 0.32),
                (out_text._x, out_text._y),
                (out_text._x, out_text._y + 0.32),
            ]
            for i in range(len(lis_txt)):
                newText = dict(
                    x=lis_pos[i][0],
                    y=lis_pos[i][1],
                    text=lis_txt[i],
                    kw=lis_kwa[i],
                )
                more1.append(newText)

            carr = [0.285, 0.296, 0.268, 0.99]
            if (col == curr_column_length - 1) and (lin == curr_column_length - 1):
                carr = [0.172, 0.198, 0.168, 0.99]
            facecolors[position] = carr

        else:
            if percentage > 0:
                txt = "%s\n%.2f%%" % (cell_value, percentage)
            else:
                txt = ""

            out_text.set_text(txt)

            if col == lin:
                out_text.set_color("w")
                facecolors[position] = [0.361, 0.812, 0.563, 0.991]
            else:
                out_text.set_color("r")

        more.extend(more1)
        less.extend(less1)

    for item in less:
        item.remove()
    for item in more:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    ax.set_title("Confusion matrix")
    ax.set_xlabel('Actual Label')
    ax.set_ylabel('Prediction Result')
    plt.tight_layout()
    plt.show()



