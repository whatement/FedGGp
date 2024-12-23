import copy
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from src.optimizer import optimizer_opt
from src.utils import Averager, client_dataloader, server_dataloader, sum_accuracy_score



class FedGGp():
    def __init__(
            self, train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution, model, options
    ):
        self.options = options
        self.device = options.gpu
        self.train_labeled_groups = train_labeled_groups
        self.train_unlabeled_groups = train_unlabeled_groups

        self.T_base = 0.5
        
        # FedAvg
        self.model = model.to(self.device)
        self.local_model = [copy.deepcopy(self.model) for _ in range(options.num_clients)]
        self.weight_data = np.array(data_distribution).sum(axis=1)

        self.lambda_pesudo = 0.1
        self.lambda_pesudo_pd = 3
        self.labeled_w = calculate_labeled_w(train_dataset, train_labeled_groups, options.num_clients, options.num_classes)
        self.unlabeled_w = calculate_labeled_w(train_dataset, train_unlabeled_groups, options.num_clients, options.num_classes)

        self.k = 5 if options.dataset == 'cifar100' else 2  # Top-K
        self.m = 0.9 # Momentum
        self.Phi = (torch.ones((self.options.num_classes)) / self.options.num_classes).to(self.device) # PHI

        self.train_labeled_loaders = client_dataloader(train_dataset, train_labeled_groups, options.num_clients, options.batch_size)
        self.train_unlabeled_loaders = client_dataloader(train_dataset, train_unlabeled_groups, options.num_clients, options.batch_size)
        self.server_test_loader = server_dataloader(test_dataset, options.batch_size)


    def trainer(self):
        client_list = range(self.options.num_clients)
        max_acc = 0.0
        p_model_w = torch.tensor(self.labeled_w.sum(axis=0) / self.labeled_w.sum()).to(self.device)

        # warmup 
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

        # training
        for r in range(1, self.options.num_rounds + 1):
            print("Round {}:".format(r))

            model_dict = self.model.state_dict()
            client_select = random.sample(client_list, int(self.options.num_clients * self.options.ratio))
            print(client_select)
            w = self.weight_data / np.sum(self.weight_data[client_select])

            train_loss_list = []
            logits = (torch.ones((self.options.num_classes)) / self.options.num_classes).to(self.device)
            pred_class_total = np.zeros(self.options.num_classes)
            logits_num = 0
            for client in client_select:
                self.local_model[client].load_state_dict(model_dict)
                client_loss, logits_delta, iter_logits_sum, pred_class = self.client_update(model=self.local_model[client], client=client)
                train_loss_list.append(round(client_loss, 4))
                if iter_logits_sum != 0:
                    logits = logits + logits_delta
                    logits_num = logits_num + iter_logits_sum
                pred_class_total += pred_class
            print(pred_class_total)

            self.server_update(client_models=self.local_model, global_model=self.model, select=client_select, w=w)

            logits = logits / logits_num
            p_model_w = torch.tensor( pred_class_total / pred_class_total.sum() - 1 / self.options.num_classes ).to(self.device)
            self.update_p_model(logits_avg=logits, w=p_model_w.clamp(min=1e-8))

            if r >= self.options.num_rounds - self.options.test_rounds:
                if (r - 1) % self.options.test_interval == 0:
                    matrix, server_accuracy, val_loss = self.test(model=self.model, loader=self.server_test_loader)
                    if max_acc < server_accuracy:
                        max_acc  = server_accuracy
                        best_model = self.model.state_dict()
                        best_round = r
                    print(" Round Acc: {:.4f}, Max Acc: {:.4f}, Train Loss: {:.4f}, Val Loss: {:.4f} ".format(server_accuracy, max_acc, np.mean(train_loss_list), val_loss))

        
    def client_update(self, model, client):
        labeled = client in self.train_labeled_groups
        unlabeled = client in self.train_unlabeled_groups
        batch_size = self.options.batch_size

        model.train()
        optimizer = optimizer_opt(model.parameters(), self.options)
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        logits_delta = (torch.ones((self.options.num_classes)) / self.options.num_classes).to(self.device)
        iter_logits_sum = 0
        pred_class = np.zeros(self.options.num_classes)
        relax_sum = 0
        relax_sum_with_base_cut = 0
        
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
                    
                    inputs = torch.cat((img_x_w, img_u_w, img_u_s), dim=0).to(self.device)
                    targets_x = targets_x.to(self.device)
                    optimizer.zero_grad()

                    _, logits = model(inputs)
                    logits_x, logits_u_w, logits_u_s  = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
                    Lx = criterion(logits_x, targets_x)

                    probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)                       
                    max_probs, max_idx = probs_u_w.max(dim=-1)
                    max_probs_mean = max_probs.mean()
                    mod = self.Phi / torch.max(self.Phi, dim=-1)[0]
                   
                    strong_vector = torch.where((self.Phi >= (1.0 / self.options.num_classes)),
                                              torch.ones(self.options.num_classes).to(self.device),
                                              torch.zeros(self.options.num_classes).to(self.device))
                    strong_effect = strong_vector[max_idx] * max_probs_mean * mod[max_idx]

                    std_probs = probs_u_w.std(dim=1).mean()
                    weak_vector = torch.where((self.Phi >= (1.0 / self.options.num_classes)) | (mod < 2/self.options.num_classes),
                                              torch.zeros(self.options.num_classes).to(self.device),
                                              torch.ones(self.options.num_classes).to(self.device))
                    weak_effect = weak_vector[max_idx] * (max_probs_mean - std_probs) * mod[max_idx] 

                    tail_vector = torch.where(mod < 2/self.options.num_classes, 
                                              torch.ones(self.options.num_classes).to(self.device), 
                                              torch.zeros(self.options.num_classes).to(self.device))
                    tail_effect = tail_vector[max_idx] * (1.0 / self.options.num_classes)
                    
                    threshold = strong_effect + weak_effect + tail_effect
                    mask = max_probs.ge(threshold).float()
                    max_probs, targets_p = torch.max(probs_u_w, dim=-1)
                    Lu = (F.cross_entropy(logits_u_s, targets_p, reduction='none') * mask).mean()
                    loss = Lx + self.lambda_pesudo_pd * Lu

                    # Top-K
                    if torch.any(tail_vector):
                        mask_base = max_probs.ge(self.T_base).float()   
                        mask_hard = torch.logical_or(mask, mask_base).float() # Samples that either pass the original mask or have high confidence (but not passing the original filter)
                        mask_relax = torch.ones_like(max_probs).to(self.device) - mask_hard
                        relax_sum = relax_sum + mask_relax.sum().item()
                        tail_classes_indices = torch.nonzero(tail_vector).squeeze()
                        topk_probs, topk_indices = probs_u_w.topk(self.k, dim=-1)
                        targets_t = torch.zeros_like(mask_relax, dtype=torch.float).to(self.device)
                        class_probs = torch.zeros_like(mask_relax, dtype=torch.float).to(self.device)
                        for i in range(topk_indices.size(0)):
                            for idx, cls in enumerate(topk_indices[i]):
                                if cls in tail_classes_indices:
                                    targets_t[i] = cls
                                    class_probs[i] = topk_probs[i][idx] / topk_probs[i][0]
                                    break
                        mask_relax = mask_relax * class_probs
                        Lt = (F.cross_entropy(logits_u_s, targets_t.long(), reduction='none') * mask_relax).mean()
                        loss = loss + self.lambda_pesudo_pd * Lt

                    loss.backward()
                    optimizer.step()
                    train_loss.add(loss.item())

                    total_correct_predictions = total_correct_predictions + ((targets_p.cpu() == targets_y) * mask.cpu()).sum().item() 
                    total_targets = total_targets + len(targets_y)

                    logits_delta = logits_delta + (probs_u_w * mask.view(-1, 1)).sum(dim=0)
                    iter_logits_sum = iter_logits_sum + torch.sum(mask).item()
                    max_idx_np = max_idx.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    np.add.at(pred_class, max_idx_np[mask_np != 0], 1)      

        elif labeled:
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
            
        elif unlabeled:
            for epoch in range(self.options.num_epochs):

                total_correct_predictions = 0
                total_targets = 0

                train_loss = Averager()
                for i, ((img, img_ema), targets_y) in enumerate(self.train_unlabeled_loaders[client]):
                    inputs = torch.cat((img, img_ema), dim=0).to(self.device)
                    optimizer.zero_grad()
                    _, logits = model(inputs)
                    logits_u_w, logits_u_s = torch.split(logits, [batch_size, batch_size], dim=0)

                    probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)                       
                    max_probs, max_idx = probs_u_w.max(dim=-1)
                    max_probs_mean = max_probs.mean()
                    mod = self.Phi / torch.max(self.Phi, dim=-1)[0]

                    strong_vector = torch.where((self.Phi >= (1.0 / self.options.num_classes)),
                                              torch.ones(self.options.num_classes).to(self.device),
                                              torch.zeros(self.options.num_classes).to(self.device))
                    strong_effect = strong_vector[max_idx] * max_probs_mean * mod[max_idx]

                    std_probs = probs_u_w.std(dim=1).mean()
                    weak_vector = torch.where((self.Phi >= (1.0 / self.options.num_classes)) | (mod < 2/self.options.num_classes),
                                              torch.zeros(self.options.num_classes).to(self.device),
                                              torch.ones(self.options.num_classes).to(self.device))
                    weak_effect = weak_vector[max_idx] * (max_probs_mean - std_probs) * mod[max_idx] 

                    tail_vector = torch.where(mod < 2/self.options.num_classes, 
                                              torch.ones(self.options.num_classes).to(self.device), 
                                              torch.zeros(self.options.num_classes).to(self.device))
                    tail_effect = tail_vector[max_idx] * (1.0 / self.options.num_classes)
                    
                    threshold = strong_effect + weak_effect + tail_effect
                    mask = max_probs.ge(threshold).float()
                    max_probs, targets_p = torch.max(probs_u_w, dim=-1)
                    Lu = (F.cross_entropy(logits_u_s, targets_p, reduction='none') * mask).mean()
                    loss = self.lambda_pesudo * Lu

                    if torch.any(tail_vector):
                        mask_base = max_probs.ge(self.T_base).float()
                        mask_hard = torch.logical_or(mask, mask_base).float()
                        mask_relax = torch.ones_like(max_probs).to(self.device) - mask_hard
                        relax_sum = relax_sum + mask_relax.sum().item()
                        tail_classes_indices = torch.nonzero(tail_vector).squeeze()
                        topk_probs, topk_indices = probs_u_w.topk(self.k, dim=-1)
                        targets_t = torch.zeros_like(mask_relax, dtype=torch.float).to(self.device)
                        class_probs = torch.zeros_like(mask_relax, dtype=torch.float).to(self.device)
                        for i in range(topk_indices.size(0)):
                            for idx, cls in enumerate(topk_indices[i]):
                                if cls in tail_classes_indices:
                                    targets_t[i] = cls
                                    class_probs[i] = topk_probs[i][idx] / topk_probs[i][0]
                                    break
                        mask_relax = mask_relax * class_probs
                        Lt = (F.cross_entropy(logits_u_s, targets_t.long(), reduction='none') * mask_relax).mean()
                        loss = loss + self.lambda_pesudo_pd * Lt

                        mask_base = max_probs.ge(self.T_base).float()
                        mask_hard = torch.logical_or(mask, mask_base).float()
                        mask_with_base = torch.ones_like(max_probs).to(self.device) - mask_hard
                        relax_sum_with_base_cut = relax_sum_with_base_cut + mask_with_base.sum().item()
                        
                    loss.backward()
                    optimizer.step()
                    train_loss.add(loss.item())

                    total_correct_predictions = total_correct_predictions + ((targets_p.cpu() == targets_y) * mask.cpu()).sum().item() 
                    total_targets = total_targets + len(targets_y)
     
        else:
            raise ValueError("Error: Client {} has no available data.".format(client))
        
        return train_loss.item(), logits_delta, iter_logits_sum, pred_class

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
    

    def update_p_model(self, logits_avg, w):
        self.Phi = self.Phi * self.m + (1 - self.m) * logits_avg / torch.exp(w)

    
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
                optimizer.step()
                train_loss.add(loss.item())


def calculate_labeled_w(train_dataset, train_labeled_groups, num_clients, num_classes):
    w = []
    for i in range(num_clients):
        class_list = [ 0 for _ in range(num_classes)]
        if i in train_labeled_groups.keys():
            for index in train_labeled_groups[i]:
                class_list[train_dataset[index][1]] += 1
        w.append(class_list)
    return np.array(w)
    