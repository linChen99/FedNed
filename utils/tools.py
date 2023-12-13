import numpy as np
import matplotlib.pyplot as plt
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def draw_plt(acc_list, loss_list):
    '''
    做出训练中的精度和损失曲线
    :param acc_list:
    :param loss_list:
    :return:
    '''
    acc_list = np.array(acc_list)
    loss_list = np.array(loss_list)
    plt.plot(acc_list)
    plt.legend(['acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    figure_save_path = "file_fig"
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
    plt.savefig(os.path.join(figure_save_path, 'acc_noniid_0.7_beta.png'))  # 第一个是指存储路径，第二个是图片名字
    # plt.savefig('acc')
    plt.cla()
    plt.plot(loss_list)
    plt.legend(['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('loss')
    plt.savefig(os.path.join(figure_save_path, 'loss_test_avg_bad.png'))#第一个是指存储路径，第二个是图片名字


def get_set_gpus(gpu_ids):
    # get gpu ids
    # if len(gpu_ids) > 1:
    #     str_ids = gpu_ids.split(',')
    #     gpus = []
    #     for str_id in str_ids:
    #         id = int(str_id)
    #         if id >= 0:
    #             gpus.append(id)
    # else:
    # gpus = [int(gpu_ids)]
    # set gpu ids
    # if len(gpus) > 0:
    # torch.cuda.set_device(gpus[0])
    return gpu_ids