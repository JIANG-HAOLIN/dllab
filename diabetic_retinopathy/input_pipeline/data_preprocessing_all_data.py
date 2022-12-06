import torch
import cv2
import PIL.Image as Image
import os
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as tf
from torchvision.utils import save_image

path_train_dataset = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/kaggle_train"
path_test_dataset = "/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/test"
# files = os.listdir(path)

h1, h2 = 728, 1024
# path = os.path.join(r"/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/train/IDRiD_001.jpg")
path_output_train = "../kaggle_preprocessed_train"
path_output_test = "../IDRID_dataset_preprocessed_test"

def image_preprocess(input_path=None,output_path=None,output_size=None):
    output_path_0 = output_path + "_original_size"
    output_path_1 = output_path + '_size%dx%d' % (output_size[0], output_size[1])
    output_path_2 = output_path + '_size1024x1024'
    output_path_3 = output_path + '_size512x512'

    if not os.path.exists(output_path_0):
        os.makedirs(output_path_0)
    if not os.path.exists(output_path_1):
        os.makedirs(output_path_1)
    if not os.path.exists(output_path_2):
        os.makedirs(output_path_2)
    if not os.path.exists(output_path_3):
        os.makedirs(output_path_3)

    # images = []

    for file_name in os.listdir(input_path):
        if file_name != '.DS_Store':
            # images.append((file, input_path, output_path, output_size))

    # for n_sample in np.linspace(1,413,413):
    #     if n_sample <= 9:
    #         n_sample = "00%d" % n_sample
    #     elif n_sample>=10 and n_sample <=99:
    #         n_sample = "0%d" % n_sample
    #     else :
    #         n_sample = "%d" % n_sample
    #
    #     path1 = input_path + "/" + "IDRiD_%s.jpg" % n_sample
    # for file in images:
            img = Image.open(input_path+'/'+file_name)
            # img = img.resize([w, h])
            img = np.array(img)
            img2= np.flip(img,1)
            img_red = img[:, :, 0]
            img_red2 = np.flip(img_red, 1)

            img_transpose = np.transpose(img,(1,0,2))
            img_transpose2 = np.flip(img_transpose,1)

            width = img.shape[1]
            height = img.shape[0]

            res = width
            res2 = width

            res_transpose = height
            res2_transpose = height

            for i in range(int(0.4*img.shape[0]),int(0.6*img.shape[0])):
                for j in range(img.shape[1]):
                    if (img[i][j][0]+img[i][j][1]+img[i][j][2]) > 40 and j < res:
                      res = j
                      break
                if j >= res:
                    break

            for i in range(int(0.4*img.shape[0]),int(0.6*img.shape[0])):
                for j in range(img.shape[1]):
                    if (img2[i][j][0]+img2[i][j][1]+img2[i][j][2]) > 40 and j < res2:
                      res2 = j
                      break
                if j >= res:
                    break

            for i in range(int(0.4*img_transpose.shape[0]),int(0.6*img_transpose.shape[0])):
                for j in range(img_transpose.shape[1]):
                    if (img_transpose[i][j][0]+img_transpose[i][j][1]+img_transpose[i][j][2]) > 40 and j < res_transpose:
                      res_transpose = j
                      break
                if j >= res_transpose:
                    break

            for i in range(int(0.4*img_transpose.shape[0]),int(0.6*img_transpose.shape[0])):
                for j in range(img_transpose2.shape[1]):
                    if (img_transpose2[i][j][0]+img_transpose2[i][j][1]+img_transpose2[i][j][2]) > 40 and j < res2_transpose:
                      res2_transpose = j
                      break
                if j >= res2_transpose:
                    break



            a, b ,c ,d = res, width - res2, res_transpose, height-res2_transpose
            if (b-a)*(d-c) >0 :
                img = img[c:d, a:b, :]
            # plt.imshow(img)
            # plt.show()
                img = np.transpose(img, (2, 0, 1))
                img = torch.tensor(img)
                img = img/255.

                img1 = tf.Resize([h1, h1])(img)
                img2 = tf.Resize([1024, 1024])(img)
                img3 = tf.Resize([512, 512])(img)
                # save_image(img, output_train+ "/" + "IDRiD_%s.jpg" % n_sample)
                save_image(img, output_path_0 +'/'+file_name)
                save_image(img1, output_path_1+'/'+file_name)
                save_image(img2, output_path_2 + '/' + file_name)
                save_image(img3, output_path_3 + '/' + file_name)
            else:
                print(file_name)



if __name__ == "__main__":
    image_preprocess(input_path=path_train_dataset,
                     output_path=path_output_train,
                     output_size=(h1, h1))
    # image_preprocess(input_path="/Users/hlj/Documents/NoSync.nosync/DL_Lab/IDRID_dataset/images/test",
    #                  output_path="../IDRID_dataset_preprocessed_test",
    #                  output_size=(h1, h1))






