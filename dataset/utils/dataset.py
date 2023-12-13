from torch.utils.data.dataset import Dataset
import numpy as np
from queue import Queue

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        print(f'{client}: {nums_data}')


#蒸馏数据集处理
def partition_train_teach(list_label2indices: list, num_data_train: int, seed=None):
    random_state = np.random.RandomState(seed)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[num_data_train // 10:])
    return list_label2indices_teach


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None
        self.queues_list = [Queue(maxsize=5) for _ in range(len(dataset))]

    def load(self, indices: list, label_list: list):
        self.indices = indices
        self.label_list = label_list

    def __getitem__(self, idx):
        index = self.indices[idx]
        image, _ = self.dataset[index]
        label = self.label_list[idx]
        return image, label,idx

    def __len__(self):
        return len(self.indices)



class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.queues_list = [Queue(maxsize=5) for _ in range(len(dataset))]

    def __getitem__(self, idx):
        image, label,index = self.dataset[idx]
        return image, label, index

    def __len__(self):
        return len(self.dataset)

    def put_queue(self, i, label):
        self.queues_list[i].put(label)

    def get_queue(self, i):
        return self.queues_list[i]

