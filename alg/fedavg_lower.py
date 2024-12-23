import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.optimizer import optimizer_opt
from src.utils import Averager, client_dataloader, server_dataloader, sum_accuracy_score


class FedAvg_Lower():
    def __init__(
            self, train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu
        self.train_labeled_groups = train_labeled_groups
        self.train_unlabeled_groups = train_unlabeled_groups
        
        # FedAvg
        self.model = model.to(self.device)
        self.client_list = list(self.train_labeled_groups.keys())
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]
        self.max_grad_norm = 5.0

        total_length = sum(len(group) for group in train_labeled_groups.values())
        self.lengths_dict = {key: len(value) / total_length for key, value in train_labeled_groups.items()}


        self.train_labeled_loaders = client_dataloader(train_dataset, train_labeled_groups, options.num_clients, options.batch_size)
        self.train_unlabeled_loaders = client_dataloader(train_dataset, train_unlabeled_groups, options.num_clients, options.batch_size)
        self.server_test_loader = server_dataloader(test_dataset, options.batch_size)


    def trainer(self):
        max_acc = 0.0
        
        for r in range(1, self.options.num_rounds + 1):
            print("Round {}:".format(r))

            model_dict = self.model.state_dict()

            # Update
            train_loss_list = []
            for client in self.client_list:
                self.local_model[client].load_state_dict(model_dict)
                client_loss = self.client_update(model=self.local_model[client], client=client, r=r)
                train_loss_list.append(round(client_loss, 4))
            print(train_loss_list)
            
            self.server_update(client_models=self.local_model, global_model=self.model)

            if r >= self.options.num_rounds - self.options.test_rounds:
                if (r - 1) % self.options.test_interval == 0:
                    matrix, server_accuracy, val_loss = self.test(model=self.model, loader=self.server_test_loader)
                    if max_acc < server_accuracy:
                        max_acc  = server_accuracy
                        best_model = self.model.state_dict()
                        best_round = r
                    print(" Round Acc: {:.4f}, Max Acc: {:.4f}, Train Loss: {:.4f}, Val Loss: {:.4f} ".format(server_accuracy, max_acc, np.mean(train_loss_list), val_loss))


    def client_update(self, model, client, r):

        model.train()
        optimizer = optimizer_opt(model.parameters(), self.options)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)

        for epoch in range(self.options.num_epochs):
            train_loss = Averager()
            for i, ((img, img_ema), target) in enumerate(self.train_labeled_loaders[client]):
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                _, logits = model(img)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                train_loss.add(loss.item())

        return train_loss.item()

    def server_update(self, client_models, global_model):
        mean_weight_dict = {}
        for name, param in global_model.state_dict().items():
            weight = []
            for index in self.client_list:
                weight.append(client_models[index].state_dict()[name] * self.lengths_dict[index])
            weight = torch.stack(weight, dim=0)
            mean_weight_dict[name] = weight.sum(dim=0)
        global_model.load_state_dict(mean_weight_dict, strict=False)
    

    def test(self, model, loader):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        predict_y = []
        img_y = []
        model.to(self.device)
        val_loss = Averager()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                _, logits = model(x)
                loss = criterion(logits, y)
                predicted = logits.argmax(dim=1)
                predict_y.extend(predicted.cpu())
                img_y.extend(y.cpu())
                val_loss.add(loss.item())

        matrix, accuracy = sum_accuracy_score(predict_y, img_y)

        return matrix, accuracy, val_loss.item()





