import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.optimizer import optimizer_opt
from src.utils import Averager, client_dataloader, server_dataloader, sum_accuracy_score


class Twin_sight():
    def __init__(
            self, train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu
        self.train_labeled_groups = train_labeled_groups
        self.train_unlabeled_groups = train_unlabeled_groups

        self.model = model.to(self.device)
        self.unsup_model = SimCLRModel(model).to(self.device)
        self.max_grad_norm = 5.0
        
        # FedAvg Twin
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]
        self.local_unsup_model = [copy.deepcopy(self.unsup_model) for _ in range(options.num_clients)]
        self.weight_data = np.array(data_distribution).sum(axis=1)

        # FreeMatch
        self.p_model = [(torch.ones((self.options.num_classes)) / self.options.num_classes).to(self.device) for _ in range(options.num_clients)]
        self.time_p = [((torch.ones((self.options.num_classes)) / self.options.num_classes).mean()).to(self.device) for _ in range(options.num_clients)]
        self.lambda_pesudo = 0.5
        self.lambda_pesudo_pd = 3
        
        # Twin Loss
        self.insDis = 0.5
        self.twin = 1.0

        self.train_labeled_loaders = client_dataloader(train_dataset, train_labeled_groups, options.num_clients, options.batch_size)
        self.train_unlabeled_loaders = client_dataloader(train_dataset, train_unlabeled_groups, options.num_clients, options.batch_size)
        self.server_test_loader = server_dataloader(test_dataset, options.batch_size)

    def trainer(self):
        client_list = range(self.options.num_clients)
        max_acc = 0.0

        # Test
        if self.options.warmup:
            print("*************Warmup************")
            labeled_lsit = []
            labeled_w_list = [0 for _ in range(self.options.num_clients)]
            total_length = sum(len(group) for group in self.train_labeled_groups.values())
            for key, value in self.train_labeled_groups.items():
                labeled_lsit.append(int(key))
                labeled_w_list[key] = len(value) / total_length

            for r in range(1, self.options.warmup_round + 1):
                model_dict = self.model.state_dict()
                for client in labeled_lsit:
                    self.local_model[client].load_state_dict(model_dict)
                    self.client_update_warmup(model=self.local_model[client], client=client)
                
                self.server_update(client_models=self.local_model, global_model=self.model, select=labeled_lsit, w=labeled_w_list)
        
            self.unsup_model = SimCLRModel(self.model).to(self.device)
            self.local_unsup_model = [copy.deepcopy(self.unsup_model) for _ in range(self.options.num_clients)]

        # 训练 round
        for r in range(1, self.options.num_rounds + 1):
            print("Round {}:".format(r))

            model_dict = self.model.state_dict()
            model_unsup_dict = self.unsup_model.state_dict()
            client_select = random.sample(client_list, int(self.options.num_clients * self.options.ratio))
            w = self.weight_data / np.sum(self.weight_data[client_select])

            # Update
            train_loss_list = []
            for client in client_select:
                self.local_model[client].load_state_dict(model_dict)
                self.local_unsup_model[client].load_state_dict(model_unsup_dict)
                client_loss = self.client_update(model=self.local_model[client], unsup_model=self.local_unsup_model[client], client=client)
                train_loss_list.append(round(client_loss, 4))
            print(train_loss_list)

            self.server_update(client_models=self.local_model, global_model=self.model, select=client_select, w=w)
            self.server_update(client_models=self.local_unsup_model, global_model=self.unsup_model, select=client_select, w=w)

            # lr decay
            if self.options.lr_decay:
                if r % 1 == 0:
                    self.options.lr = self.options.lr * 0.99

            if r >= self.options.num_rounds - self.options.test_rounds:
                if (r - 1) % self.options.test_interval == 0:
                    matrix, server_accuracy, val_loss = self.test(model=self.model, loader=self.server_test_loader)
                    if max_acc < server_accuracy:
                        max_acc  = server_accuracy
                        best_model = self.model.state_dict()
                        best_round = r
                    print(" Round Acc: {:.4f}, Max Acc: {:.4f}, Train Loss: {:.4f}, Val Loss: {:.4f} ".format(server_accuracy, max_acc, np.mean(train_loss_list), val_loss))


    def client_update(self, model, unsup_model, client):
        labeled = client in self.train_labeled_groups
        unlabeled = client in self.train_unlabeled_groups
        batch_size = self.options.batch_size

        model.train()
        unsup_model.train()
        optimizer = optimizer_opt(model.parameters(), self.options)
        optimizer_un = optimizer_opt(unsup_model.parameters(), self.options)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        unsup_model = unsup_model.to(self.device)

        if labeled and unlabeled:
            labeled_iter = iter(self.train_labeled_loaders[client])
            unlabeled_iter = iter(self.train_unlabeled_loaders[client])
            batch_len = max(len(self.train_labeled_loaders[client]), len(self.train_unlabeled_loaders[client]))
            
            for epoch in range(self.options.num_epochs):
                
                total_correct_predictions = 0
                total_targets = 0

                train_loss = Averager()
                for i in range(batch_len):
                    try:
                        (img_x_w, img_x_s), targets_x = labeled_iter.next()
                    except:
                        labeled_iter = iter(self.train_labeled_loaders[client])
                        (img_x_w, img_x_s), targets_x = labeled_iter.next()
                        
                    try:
                        (img_u_w, img_u_s), targets_y = unlabeled_iter.next() # weak,strong
                    except:
                        unlabeled_iter = iter(self.train_unlabeled_loaders[client])
                        (img_u_w, img_u_s), targets_y = unlabeled_iter.next()

                    optimizer.zero_grad()
                    optimizer_un.zero_grad()
                    img_x_w = img_x_w.to(self.device)
                    img_x_s = img_x_s.to(self.device)
                    img_u_w = img_u_w.to(self.device)
                    img_u_s = img_u_s.to(self.device)
                    targets_x = targets_x.to(self.device)

                    ###### Supervised Loss ###### 
                    inputs = torch.cat((img_x_w, img_x_s, img_u_w, img_u_s), dim=0)
                    f, logits = model(inputs)
                    f_x_w, f_x_s, f_u_w, f_u_s  = torch.split(f, [batch_size, batch_size, batch_size, batch_size], dim=0)
                    logits_x_w, logits_x_s, logits_u_w, logits_u_s  = torch.split(logits, [batch_size, batch_size, batch_size, batch_size], dim=0)
                    Lx = criterion(logits_x_w, targets_x) 
                    probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)

                    self.adaptive_local_threshold(probs_u_w, client)
                    max_probs, max_idx = probs_u_w.max(dim=-1)
                    mod = self.p_model[client] / torch.max(self.p_model[client], dim=-1)[0]
                    max_probs, targets_p = torch.max(probs_u_w, dim=-1)
                    mask = max_probs.ge(self.time_p[client] * mod[max_idx]).float()

                    Lu = (F.cross_entropy(logits_u_s, targets_p, reduction='none') * mask).mean()
                    ce_loss = Lx + self.lambda_pesudo_pd * Lu

                    ###### Unsupervised Loss ###### 
                    inputs_CLR = torch.cat((img_x_w, img_u_w, img_x_s, img_u_s), dim=0)
                    unf, un_logits = unsup_model(inputs_CLR)
                    CLR_logits, CLR_labels = info_nce_loss(un_logits, batch_size * 2, self.device)
                    insDis_loss = criterion(CLR_logits, CLR_labels)
                    unf_x_w, unf_u_w, unf_x_s, unf_u_s  = torch.split(unf, [batch_size, batch_size, batch_size, batch_size], dim=0)

                    ###### Twin Sight Loss ###### 
                    l_x = TwinLoss(f_x_w, unf_x_w)
                    l_u = TwinLoss(f_u_w, unf_u_w)
                    twin_loss = 0.5 * (l_x + l_u)

                    loss = ce_loss + self.insDis * insDis_loss + self.twin * twin_loss
                    
                    loss.backward()
                    optimizer.step()
                    optimizer_un.step()

                    train_loss.add(loss.item())

                    total_correct_predictions = total_correct_predictions + ((targets_p.cpu() == targets_y) * mask.cpu()).sum().item() 
                    total_targets = total_targets + len(targets_y)

            print(" Client: {}, Correct Predictions: {}, Total Targets: {} ".format(client, total_correct_predictions, total_targets))
            
        elif labeled:
            for epoch in range(self.options.num_epochs):
                train_loss = Averager()
                for i, ((img, img_ema), target) in enumerate(self.train_labeled_loaders[client]):
                    img = img.to(self.device)
                    img_ema = img_ema.to(self.device)
                    target = target.to(self.device)
                    optimizer.zero_grad()
                    optimizer_un.zero_grad()

                    ###### Supervised Loss ###### 
                    inputs = torch.cat((img, img_ema), dim=0)
                    f, logits = model(inputs)
                    f_x_w, f_x_s = torch.split(f, [batch_size, batch_size], dim=0)
                    logits_x_w, logits_x_s = torch.split(logits, [batch_size, batch_size], dim=0)
                    Lx = criterion(logits_x_w, target) 
                    ce_loss = Lx

                    ###### Unsupervised Loss ###### 
                    unf, un_logits = unsup_model(inputs)
                    CLR_logits, CLR_labels = info_nce_loss(un_logits, batch_size, self.device)
                    insDis_loss = criterion(CLR_logits, CLR_labels)
                    unf_x_w, unf_x_s = torch.split(unf, [batch_size, batch_size], dim=0)

                    ###### Twin Sight Loss ###### 
                    l_x = TwinLoss(f_x_w, unf_x_w)
                    twin_loss = l_x

                    # print(ce_loss, insDis_loss, twin_loss)
                    loss = ce_loss + self.insDis * insDis_loss + self.twin * twin_loss

                    loss.backward()
                    optimizer.step()
                    optimizer_un.step()
                    train_loss.add(loss.item())
            
        elif unlabeled:
            for epoch in range(self.options.num_epochs):
                
                total_correct_predictions = 0
                total_targets = 0

                train_loss = Averager()
                for i, ((img, img_ema), targets_y) in enumerate(self.train_unlabeled_loaders[client]):
                    img = img.to(self.device)
                    img_ema = img_ema.to(self.device)
                    optimizer.zero_grad()
                    optimizer_un.zero_grad()

                    ###### Supervised Loss ###### 
                    inputs = torch.cat((img, img_ema), dim=0)
                    f, logits = model(inputs)
                    f_u_w, f_u_s = torch.split(f, [batch_size, batch_size], dim=0)
                    logits_u_w, logits_u_s = torch.split(logits, [batch_size, batch_size], dim=0)
                    probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)

                    self.adaptive_local_threshold(probs_u_w, client)
                    max_probs, max_idx = probs_u_w.max(dim=-1)
                    mod = self.p_model[client] / torch.max(self.p_model[client], dim=-1)[0]
                    max_probs, targets_u = torch.max(probs_u_w, dim=-1)
                    mask = max_probs.ge(self.time_p[client] * mod[max_idx]).float()

                    Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
                    ce_loss = self.lambda_pesudo * Lu

                    ###### Unsupervised Loss ###### 
                    unf, un_logits = unsup_model(inputs)
                    CLR_logits, CLR_labels = info_nce_loss(un_logits, batch_size, self.device)
                    insDis_loss = criterion(CLR_logits, CLR_labels)
                    unf_u_w, unf_u_s = torch.split(unf, [batch_size, batch_size], dim=0)

                    ###### Twin Sight Loss ###### 
                    l_u = TwinLoss(f_u_w, unf_u_w)
                    twin_loss = l_u

                    loss = ce_loss + self.insDis * insDis_loss + self.twin * twin_loss
                    
                    loss.backward()
                    optimizer.step()
                    optimizer_un.step()

                    train_loss.add(loss.item())

                    total_correct_predictions = total_correct_predictions + ((targets_u.cpu() == targets_y) * mask.cpu()).sum().item() 
                    total_targets = total_targets + len(targets_y)

            print(" Client: {}, Correct Predictions: {}, Total Targets: {} ".format(client, total_correct_predictions, total_targets))

        else:
            raise ValueError("Error: Client {} has no available data.".format(client))


        return train_loss.item()

    def server_update(self, client_models, global_model, select, w):
        mean_weight_dict = {}
        for name, param in global_model.state_dict().items():
            weight = []
            for index in select:
                weight.append(client_models[index].state_dict()[name] * w[index])
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
    
    def adaptive_local_threshold(self, probs_u_w, client):
        max_probs, _ = torch.max(probs_u_w, dim=-1, keepdim=True)
        self.time_p[client] = self.time_p[client] * 0.999 + 0.001 * max_probs.mean()
        self.p_model[client] = self.p_model[client] * 0.999 + 0.001 * probs_u_w.mean(dim=0)

    def client_update_warmup(self, model, client):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                train_loss.add(loss.item())



class SimCLRModel(nn.Module):
    def __init__(self, net, ):
        super().__init__()

        self.backbone = copy.deepcopy(net)
        dim_mlp = self.backbone.feature_dim
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp * 2), nn.BatchNorm1d(dim_mlp * 2), nn.ReLU(), nn.Linear(dim_mlp * 2, dim_mlp))

    def forward(self, image):
        return self.backbone(image)


def info_nce_loss( features, batch_size, device, n_views=2, temperature=0.07):
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def TwinLoss(x, y):
    x = F.normalize(x, dim=1)
    x_matrix = torch.mm(x, x.t())
    y = F.normalize(y, dim=1)
    y_matrix = torch.mm(y, y.t())
    l = torch.sum(torch.pow((x_matrix - y_matrix), 2) / (x.size(0) ** 2))
    return l