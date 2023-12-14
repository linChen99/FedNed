import copy
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch import stack, no_grad
from torch.optim import SGD, Adam, lr_scheduler
from models.model_feature import ResNet_cifar_feature
from client import Client
from torch.nn.functional import softmax, log_softmax
import torch
import numpy as np
import torch.nn as nn



class Server(object):
    def __init__(self,
                 args,
                 train_data_list,
                 global_test_dataset,
                 global_distill_dataset,
                 global_student,
                 temperature: float,
                 mini_batch_size_distillation: int,
                 lamda
):

        super(Server, self).__init__()
        self.device = args.device
        self.global_rounds = args.global_rounds
        self.batch_size = args.batch_size
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.lamda =lamda
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.train_data_list = train_data_list
        self.global_test_dataset = global_test_dataset
        self.global_distill_dataset = global_distill_dataset
        # self.global_teaching_dataset = global_teaching_dataset
        self.global_student = copy.deepcopy(global_student)
        self.global_student.to(self.device)
        self.dict_global_params = self.global_student.state_dict()
        self.optimizer = Adam(self.global_student.parameters(), lr=args.global_learning_rate, weight_decay=0.002)

        #teacher model
        self.global_teacher_good = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                                        save_activations=False, group_norm_num_groups=None,
                                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.global_teacher_good.to(self.device)
        self.global_teacher_bad = ResNet_cifar_feature(resnet_size=8, scaling=4,
                                                       save_activations=False, group_norm_num_groups=None,
                                                       freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes)
        self.global_teacher_bad.to(self.device)
        self.dict_global_bad_teacher_params = self.global_teacher_bad.state_dict()


        self.clients = []
        self.clients_bad = []
        self.list_noisy_ratio = args.noise_rate_list
        self.list_dicts_local_params = []
        self.list_nums_local_data = []
        self.list_dicts_good_local_params = []
        self.list_nums_good_local_data = []
        self.list_dicts_bad_local_params = []
        self.list_nums_bad_local_data = []
        #loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.test_acc = []
        self.test_loss = []
        #distill
        self.temperature = temperature
        self.mini_batch_size_distillation = mini_batch_size_distillation
        self.random_state = np.random.RandomState(args.seed)
        self.set_clients(args, Client)
        self.warm_round = 10

    def set_clients(self, args, FedAvg):
        for i in range(self.num_clients):
            client = FedAvg(
                args=args,
                data_client=self.train_data_list[i],
                train_sapmles=len(self.train_data_list[i]),
                model=copy.deepcopy(self.global_student),
            )
            self.clients.append(client)
            self.clients_bad.append(client)

    def select_clinet_indexes(self):
        selected_client_indexes = list(
            np.random.choice(list(range(self.num_clients)), self.join_clients, replace=False))
        return selected_client_indexes

    def send_models(self):
        assert (len(self.selected_client_indexes) > 0)
        for select_id in self.selected_client_indexes:
            self.clients[select_id].download_params(copy.deepcopy(self.dict_global_params))

    def aggregate_parameters(self):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(self.list_dicts_good_local_params, self.list_nums_good_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)

            value_global_param = sum(list_values_param) / sum( self.list_nums_good_local_data)
            self.dict_global_params[name_param] = value_global_param
        self.global_teacher_good.load_state_dict(self.dict_global_params)

    def aggregate_wram_parameters(self):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(self.list_dicts_local_params, self.list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)

            value_global_param = sum(list_values_param) / sum(self.list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param
        self.global_teacher_good.load_state_dict(self.dict_global_params)

    def get_bad_logit(self, images, list_dicts_bad_local_params):
        list_logits = []
        list_softmax = []
        for dict_local_params in list_dicts_bad_local_params:
            self.global_teacher_bad.load_state_dict(dict_local_params)
            self.global_teacher_bad.eval()
            with no_grad():
                _, _, local_logits = self.global_teacher_bad(images)
                local_logits = torch.reciprocal(local_logits)
                local_softmax = softmax(local_logits, dim=1)
                list_logits.append(copy.deepcopy(local_logits))
                list_softmax.append(local_softmax)
        return list_softmax

    def compute_uncertainty(self):
        client_indexes = self.selected_client_indexes
        list_dicts_local_params = self.list_dicts_local_params
        total_indices_unlabeled = [i for i in range(len(self.global_distill_dataset))]
        batch_indices_unlabeled = self.random_state.choice(total_indices_unlabeled,
                                                           self.mini_batch_size_distillation,
                                                           replace=False)
        images_unlabeled = []
        for idx in batch_indices_unlabeled:
            image, _ = self.global_distill_dataset[idx]
            images_unlabeled.append(image)
        images_unlabeled = stack(images_unlabeled, dim=0)
        images_unlabeled = images_unlabeled.to(self.device)
        list_var = {}
        list_key_good = []
        list_key_bad = []
        list_softmax = []
        ii = 0
        for dict_local_params in list_dicts_local_params:
            for name_param in self.dict_global_params:
                self.dict_global_params[name_param] = dict_local_params[name_param]
            self.global_student.load_state_dict(self.dict_global_params)

            uncen = torch.tensor([]).cuda()
            pred_prob = torch.tensor([]).cuda()
            for step in range(10):
                with no_grad():
                    _, local_drop, local_logits = self.global_student(images_unlabeled)
                    drop_softmax = softmax(local_drop, dim=1).view(-1, 10, 1)
                    pred_prob = torch.cat([pred_prob, drop_softmax], dim=-1)
            list_softmax.append(pred_prob.reshape(-1).cpu().detach().numpy())

            uncen = pred_prob.mean(-1)
            per_uncen = -torch.sum(torch.log(uncen) * uncen, dim=-1)
            list_var[client_indexes[ii]] = torch.sum(per_uncen)
            ii = ii + 1

        list_var = sorted(list_var.items(), key=lambda x:x[1], reverse=False)

        sum_value = 0
        for key, value in list_var:
            sum_value = sum_value + value

        for key, value in list_var:
            value = value / sum_value
            if value <= self.lamda:
                list_key_good.append(key)
            else:
                list_key_bad.append(key)

        print('list_key_good:{}'.format(list_key_good))
        print('list_key_bad:{}'.format(list_key_bad))
        return list_key_good,list_key_bad


    def aggregate_distillation(self):
        self.aggregate_parameters()
        # self.aggregate_wram_parameters()
        for step in range(100):
            total_indices_unlabeled = [i for i in range(len(self.global_distill_dataset))]
            batch_indices_unlabeled = self.random_state.choice(total_indices_unlabeled,
                                                               self.mini_batch_size_distillation,
                                                               replace=False)
            images_unlabeled = []
            for idx in batch_indices_unlabeled:
                image, _ = self.global_distill_dataset[idx]
                images_unlabeled.append(image)
            images_unlabeled = stack(images_unlabeled, dim=0)
            images_unlabeled = images_unlabeled.to(self.device)

            total_logits_bad_teacher = self.get_bad_logit(images_unlabeled, self.list_dicts_bad_local_params)
            _, _, avg_logits_good_teacher = self.global_teacher_good(images_unlabeled)
            y_good = softmax(avg_logits_good_teacher / self.temperature, dim=1)
            loss = 0
            if len(total_logits_bad_teacher) > 0:
                for bad_logit in total_logits_bad_teacher:
                    y_bad = (bad_logit / self.temperature).to(self.device)
                    bad_teacher_loss = F.kl_div(y_good.log(), y_bad, reduction='batchmean')
                    loss = loss + bad_teacher_loss
            else:
                y_bad = torch.zeros((128, 10)).to(self.device)
                loss = F.kl_div(y_good.log(), y_bad, reduction='batchmean')

            loss = loss / len(total_logits_bad_teacher)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.dict_global_params = self.global_teacher_good.state_dict()

    def train(self):
        bad_log = []
        for i in range(1, self.global_rounds + 1):
            # warm_up
            if i <= self.warm_round:
                self.selected_client_indexes = self.select_clinet_indexes()
                print('selected_client_indexes: {}'.format(self.selected_client_indexes))
                self.send_models()
                for select_id in self.selected_client_indexes:
                    self.clients[select_id].train()
                    self.list_dicts_local_params.append(self.clients[select_id].upload_params())
                    self.list_nums_local_data.append(self.clients[select_id].train_samples)

                self.aggregate_wram_parameters()
                self.list_dicts_local_params = []
                self.list_nums_local_data = []
                acc, loss = self.evaluate()
                print('Round: {}'.format(i))
                print('{:.5f}'.format(acc))
            else:
                self.selected_client_indexes= self.select_clinet_indexes()
                print('selected_client_indexes: {}'.format(self.selected_client_indexes))
                self.send_models()
                print('EN indax: {}'.format(bad_log))
                for select_id in self.selected_client_indexes:
                    if select_id in bad_log:
                        self.clients[select_id].bad_train()
                        self.clients_bad[select_id].train()
                        self.list_dicts_local_params.append(self.clients[select_id].upload_params())
                        self.list_nums_local_data.append(self.clients[select_id].train_samples)

                    else:
                        self.clients[select_id].train()
                        self.list_dicts_local_params.append(self.clients[select_id].upload_params())
                        self.list_nums_local_data.append(self.clients[select_id].train_samples)
                selected_good_client_indexes, selected_bad_client_indexes = self.compute_uncertainty()

                self.selected_good_client_indexes = selected_good_client_indexes
                self.selected_bad_client_indexes = selected_bad_client_indexes

                for id in selected_bad_client_indexes:
                    if id not in bad_log:
                        bad_log.append(id)

                for select_id in self.selected_good_client_indexes:
                    self.list_dicts_good_local_params.append(self.clients[select_id].upload_params())
                    self.list_nums_good_local_data.append(self.clients[select_id].train_samples)

                for select_id in self.selected_bad_client_indexes:
                    self.list_dicts_bad_local_params.append(self.clients[select_id].upload_params())
                    self.list_nums_bad_local_data.append(self.clients[select_id].train_samples)

                for select_id in bad_log:
                    if select_id in self.selected_client_indexes and select_id not in self.selected_bad_client_indexes:
                        self.list_dicts_bad_local_params.append(self.clients_bad[select_id].upload_params())
                        self.list_nums_bad_local_data.append(self.clients_bad[select_id].train_samples)


                print('good_client: {}'.format(self.selected_good_client_indexes))
                print('bad_client: {}'.format(self.selected_bad_client_indexes))

                self.aggregate_distillation()
                self.list_dicts_local_params = []
                self.list_nums_local_data = []
                self.list_dicts_good_local_params = []
                self.list_nums_good_local_data = []
                self.list_dicts_bad_local_params = []
                self.list_nums_bad_local_data = []

                acc, loss= self.evaluate()
                print('Round: {}'.format(i))
                print('Acc:{:.5f}'.format(acc))

    def evaluate(self):
        self.global_teacher_good.eval()

        with torch.no_grad():
            test_loader =DataLoader(self.global_test_dataset,
                       batch_size=self.batch_size,
                       shuffle=True)

            num_corrects = 0
            total_loss = 0.0

            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                feature, _, outputs = self.global_teacher_good(images)

                loss = self.ce_loss(outputs, labels)
                total_loss += loss.item()

                _, predicts = torch.max(outputs, -1)
                num_corrects += sum(torch.eq(predicts.cpu(), labels.cpu())).item()

            accuracy = num_corrects / len(self.global_test_dataset)

        self.test_acc.append(accuracy)
        self.test_loss.append(total_loss)

        return accuracy, total_loss






