import random


def noisify_label(true_label, label_lst, noise_type):

    if noise_type == "symmetric":
        result_dic = []
        for item in label_lst:
            if item not in result_dic:
                result_dic.append(item)
        label_lst = result_dic
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        if true_label == label_lst[0]:
            label = label_lst[1]
        elif true_label == label_lst[1]:
            label = label_lst[0]
        elif true_label == label_lst[2]:
            label = label_lst[3]
        elif true_label == label_lst[3]:
            label = label_lst[2]
        elif true_label == label_lst[4]:
            label = label_lst[5]
        elif true_label == label_lst[5]:
            label = label_lst[4]
        elif true_label == label_lst[6]:
            label = label_lst[7]
        elif true_label == label_lst[7]:
            label = label_lst[6]
        elif true_label == label_lst[8]:
            if len(label_lst) > 9:
                label = label_lst[9]
            else:
                label = label_lst[0]
        else:
            label = label_lst[0]
        return label
