import copy
import math
import random
import os
import json

import numpy as np
from collections import Counter

from src.utils import save_to_json, convert_keys_to_int


def split_dataset(args, train_dataset, test_dataset):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.labeled_partition == "client":
        folder_name = f"{args.dataset}_{args.num_clients}_{args.labeled_partition}_{args.labeled_clients}_{args.dirichlet}_{args.seed}"
    elif args.labeled_partition == "ratio":
        folder_name = f"{args.dataset}_{args.num_clients}_{args.labeled_partition}_{args.labeled_ratio}_{args.dirichlet}_{args.seed}"
    elif args.labeled_partition == "hybrid":
        folder_name = f"{args.dataset}_{args.num_clients}_{args.labeled_partition}_{args.labeled_clients}_{args.part_labeled_clients}_{args.labeled_ratio}_{args.dirichlet}_{args.seed}"
    else:
        raise NotImplementedError
    
    print("Splitting Datasets:")

    directory = os.path.join("./datasplit/", folder_name)
    if not os.path.exists(directory):

        train_groups_split = split_by_dirichlet(train_dataset, args.num_clients, args.num_classes, args.dirichlet)
        test_groups = split_by_iid(test_dataset, args.num_clients, args.num_classes)

        train_distribution_list = show_data_distribution(train_dataset, train_groups_split, args.num_classes)

        max_iterations = 100
        iteration_count = 0
        while(True):
            iteration_count += 1
            print("**************** iteration_count {} ****************".format(iteration_count))
            train_groups = copy.deepcopy(train_groups_split)

            train_labeled_groups, train_unlabeled_groups = {}, {}

            if args.labeled_partition == "client":
                chosen_client_indices = random.sample([i for i in range(args.num_clients)], args.labeled_clients)
                for c in chosen_client_indices:
                    train_labeled_groups[c] = train_groups.pop(c, None)
                train_unlabeled_groups = train_groups
            
            elif args.labeled_partition == "ratio":
                for c in range(args.num_clients):
                    labeled_num = max(math.ceil( len(train_groups[c]) * args.labeled_ratio), args.batch_size)
                    train_labeled_groups[c] = np.random.choice(train_groups[c], labeled_num, replace=False)
                    train_unlabeled_groups[c] = np.setdiff1d(train_groups[c], train_labeled_groups[c])
                
            elif args.labeled_partition == "hybrid":

                client_list = [i for i in range(args.num_clients)]
                chosen_client_indices = random.sample(client_list, args.labeled_clients)
                for i in chosen_client_indices:
                    train_labeled_groups[i] = train_groups.pop(i, None)

                remaining = [item for item in client_list if item not in chosen_client_indices]
                chosen_client_indices_part = random.sample(remaining, args.part_labeled_clients)
                for c in chosen_client_indices_part:
                    labeled_num = max(math.ceil( len(train_groups[c]) * args.labeled_ratio), args.batch_size)
                    train_labeled_groups[c] = np.random.choice(train_groups[c], labeled_num, replace=False)
                    train_unlabeled_groups[c] = np.setdiff1d(train_groups[c], train_labeled_groups[c])

                remaining_un = [item for item in remaining if item not in chosen_client_indices_part]
                for i in remaining_un:
                    train_unlabeled_groups[i] = train_groups.pop(i, None)

            else:
                raise NotImplementedError
            
            # class test
            all_labeled_classes = set()
            labeled_classes_client_group = {key: [] for key in train_labeled_groups}
            for group_key, group_indices in train_labeled_groups.items():
                group_key_labeled = set()
                for idx in group_indices:
                    _, target = train_dataset[idx]
                    class_label = int(target)
                    group_key_labeled.add(class_label)
                    all_labeled_classes.add(class_label)
                labeled_classes_client_group[group_key] = list(group_key_labeled)
                print(group_key, labeled_classes_client_group[group_key])
            all_classes = set(range(args.num_classes))
            missing_classes = all_classes - all_labeled_classes
            print(f"missing classes: {missing_classes}")
            if len(missing_classes) == 0 or args.split_empty:
                os.makedirs(directory)
                save_to_json(train_labeled_groups, directory, "train_labeled_groups.json")
                save_to_json(train_unlabeled_groups, directory, "train_unlabeled_groups.json")
                save_to_json(test_groups, directory, "test_groups.json")
                save_to_json(train_distribution_list, directory, "train_distribution_list.json")
                save_to_json(labeled_classes_client_group, directory, "labeled_classes_client_group.json")
                break
            
            if iteration_count >= max_iterations:
                print("Error: Exceeded the maximum number of iterations without finding a solution.")
                exit()
    
    else:
        print(f"Loading data groups from {directory}...")
        train_labeled_groups = convert_keys_to_int(json.load(open(os.path.join(directory, "train_labeled_groups.json"))))
        train_unlabeled_groups = convert_keys_to_int(json.load(open(os.path.join(directory, "train_unlabeled_groups.json"))))
        test_groups = convert_keys_to_int(json.load(open(os.path.join(directory, "test_groups.json"))))
        train_distribution_list = json.load(open(os.path.join(directory, "train_distribution_list.json")))

    return train_labeled_groups, train_unlabeled_groups, test_groups, train_distribution_list


def classify_label(dataset, num_classes: int):
    """
    :param dataset:
    :param num_classes:
    :return: label_list
    """
    label_list = [[] for _ in range(num_classes)]
    for index, datum in enumerate(dataset):
        label_list[datum[1]].append(index)
    return label_list


def show_data_distribution(dataset, dict_split: dict, num_classes):
    """
    打印分布后的 list
    :param dataset:
    :param dict_split: 
    :param num_classes:
    :return: list_per_client
    """
    list_per_client = []
    clients_indices = list(dict_split.values())
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[int(round(idx))][1]
            nums_data[label] += 1
        list_per_client.append(nums_data)
        print(f'{client + 1}: {nums_data}')

    return list_per_client


def split_by_iid(dataset, num_clients: int, num_classes: int):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_clients:
    :param num_classes:
    :return: dict of image index
    """
    dict_users = {}

    list_label_index = classify_label(dataset, num_classes)
    data_set = [[] for _ in range(num_clients)]

    for i in range(len(list_label_index)):
        num_items = int(len(list_label_index[i]) / num_clients)
        for j in range(num_clients):
            new_set = np.random.choice(list_label_index[i], num_items, replace=False)
            data_set[j].extend(new_set)
            list_label_index[i] = np.setdiff1d(list_label_index[i], new_set)

    for k in range(num_clients):
        dict_users[k] = np.array(data_set[k])

    return dict_users


def split_by_dirichlet(train_dataset, num_clients, num_classes, dirichlet_alpha):
    """
    Sample non I.I.D. client data from dataset, split by dirichlet
    :param train_dataset:
    :param num_clients:
    :para num_classes:
    :param dirichlet_alpha:
    :return: dict of image index 
    """
    train_dict_users = {}

    train_list_label_index = classify_label(train_dataset, num_classes)
    train_client_idcs = [[] for _ in range(num_clients)]

    max_attempts = 10
    attempt_count = 0

    while attempt_count < max_attempts:
        proportions = np.random.dirichlet([dirichlet_alpha] * num_clients, num_classes)
        proportions /= proportions.sum(axis=1, keepdims=True)
        
        all_clients_have_min_samples = True
        for j in range(num_clients):
            if np.sort(proportions[:, j])[-2] * len(train_list_label_index[0]) < 1:
                all_clients_have_min_samples = False

        if all_clients_have_min_samples:
            break
        else:
            attempt_count += 1
            print(f"Try to generate the Dirichlet distribution, Attempt {attempt_count}!")
    
    if attempt_count == max_attempts:
        raise RuntimeError("Failed to generate a suitable Dirichlet distribution within the maximum number of attempts.")
    
    for c, fracs in zip(train_list_label_index, proportions):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            train_client_idcs[i] += [idcs]
    train_client_idcs = [np.concatenate(idcs) for idcs in train_client_idcs]

    for k in range(num_clients):
        train_dict_users[k] = train_client_idcs[k]

    return train_dict_users
