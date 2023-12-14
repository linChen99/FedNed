import torch.nn as nn
from torch import optim
import torch
import copy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



class Client(object):
    def __init__(self, args, data_client, train_sapmles, model):
        super(Client, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.num_classes = args.num_classes
        self.eval_step = args.eval_step
        self.data_client = data_client
        self.train_samples = train_sapmles
        self.data_path = args.data_path

        self.model = model
        self.model.to(self.device)

        self.model_classifier = model.classifier
        self.model_classifier.to(self.device)

        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.temperature = args.temperature
        self.threshold =args.threshold
        self.data_path = args.data_path


    def download_params(self, global_params):
        self.model.load_state_dict(global_params)


    def upload_params(self):
        return copy.deepcopy(self.model.state_dict())


    def train(self):
        self.model.train()
        data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        for step in range(self.local_steps):
            for images, labels, indexs in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                _, _, output = self.model(images)
                loss = self.ce_loss(output, labels)
                loss.backward()
                self.optimizer.step()

    def bad_train(self):

        self.model.train()
        labeled_data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=self.batch_size,
                                 shuffle=True)

        labeled_iter = iter(labeled_data_loader)
        for step in range(self.local_steps):
            for inputs, label, indexs in labeled_data_loader:
                inputs, label = inputs.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                _, _, logits = self.model(inputs)
                pseudo_label = torch.softmax(logits.detach() / self.temperature, dim=-1)
                max_probs, targets = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.threshold).float()
                loss = (self.ce_loss(logits, targets) * mask).mean()
                loss.backward()
                self.optimizer.step()



    def get_logit(self, image):
        _, logit = self.model(image)
        return logit

    def get_params(self):
        self.model_classifier.train()
        data_loader = DataLoader(dataset=self.data_client,
                                 batch_size=self.batch_size,
                                 shuffle=True)
        for step in range(self.local_steps):
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                feature, _, _ = self.model(images)
                feature.detach()
                output = self.model_classifier(feature)
                loss = self.ce_loss(output, labels)
                loss.backward()
                self.optimizer.step()
        net_parameters = self.model_classifier.parameters()

        return net_parameters