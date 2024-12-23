import torch


def optimizer_opt(parameter, options):
    if options.optimizer == "sgd":
        optim = sgd_optimizer(parameter, options.lr, options.sgd_momentum, options.sgd_weight_decay)
    elif options.optimizer == "adam":
        optim = adam_optimizer(parameter, options.lr, options.adam_weight_decay)
    elif options.optimizer == "adamw":
        optim = adamw_optimizer(parameter, options.lr)

    else:
        raise ValueError("No such optimizer: {}".format(options.optimizer))

    return optim


def sgd_optimizer(parameter, lr, momentum, weight_decay):
    sgd = torch.optim.SGD(parameter, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return sgd


def adam_optimizer(parameter, lr, weight_decay):
    adam = torch.optim.Adam(parameter, lr=lr, weight_decay=weight_decay)
    return adam


def adamw_optimizer(parameter, lr):
    adamw = torch.optim.AdamW(parameter, lr=lr)
    return adamw
