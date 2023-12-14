import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
from server import Server
from dataset.get_cifar10 import get_cifar10
from dataset.utils.dataset import Indices2Dataset
from models.model_feature import ResNet_cifar_feature
from dataset.utils.noisify import noisify_label
from utils.tools import get_set_gpus
from options import args_parser
import numpy as np
import copy
import torch
import random

def get_train_label(data_local_training, index_list):
    trian_label_list = []
    for index in index_list:
        label = data_local_training[index][1]
        trian_label_list.append(label)
    return trian_label_list


def label_rate(test_label_list, train_label_list):
    true_num = 0
    for true_label, nos_label in zip(test_label_list, train_label_list):
        if true_label == nos_label:
            true_num += 1
    rate = true_num / len(test_label_list)
    print(rate)

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def main(args):
    prev_time = datetime.now()
    if args.gpu:
        gpus = get_set_gpus(args.gpu)
        print('==>Currently use GPU: {}'.format(gpus))

    data_local_training, data_global_test, list_client2indices,global_distill_dataset= get_cifar10(args)


    model = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                 save_activations=False, group_norm_num_groups=None,
                                 freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
    ###add noise
    train_data_list = []
    label_list = []
    alpha = args.alpha
    beta = args.beta
    beta_samples = np.random.beta(alpha, beta, size=args.num_clients)
    noise_rate_list = np.sort(beta_samples)
    args.noise_rate_list = noise_rate_list
    print(args.noise_rate_list)

    for i in range(args.num_clients):
        current_client_index_list = list_client2indices[i]
        train_label_list = get_train_label(data_local_training, current_client_index_list)
        num_classes = train_label_list
        test_label_list = copy.deepcopy(train_label_list)
        noise_index = int(len(list_client2indices[i]) * args.noise_rate_list[i])
        for idx, true_label in enumerate(train_label_list[:noise_index]):
            noisy_label = noisify_label(true_label, num_classes, noise_type=args.noise_type)
            train_label_list[idx] = noisy_label

        label_rate(test_label_list, train_label_list)
        indices2data = Indices2Dataset(data_local_training)
        data_client = indices2data
        data_client.load(list_client2indices[i], train_label_list)
        train_data_list.append(data_client)
        label_list.append(train_label_list)

    # 打印每个客户端的真实数据分布
    for client, indices in enumerate(label_list):
        nums_data = [0 for _ in range(10)]
        for idx in indices:
            # label = data_local_training[idx][1]
            nums_data[idx] += 1
        print(f'{client}: {nums_data}')

    server = Server(args=args,
                    train_data_list=train_data_list,
                    global_test_dataset=data_global_test,
                    global_distill_dataset=global_distill_dataset,
                    global_student=model,
                    temperature=args.temperature,
                    mini_batch_size_distillation=args.mini_batch_size_distillation,
                    lamda=args.lamda
                    )

    server.train()

    acc = server.test_acc
    acc.sort()
    acc = acc[90:]
    print('train finished---> final_acc={}'.format(sum(acc) / len(acc)))

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "train time %02d:%02d:%02d" % (h, m, s)
    print(time_str)


if __name__ == '__main__':
    set_seed(0)
    args = args_parser()
    main(args)

