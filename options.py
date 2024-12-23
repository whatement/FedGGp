import argparse
import os



def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    # General setup
    parser.add_argument('--gpu', type=str, default='cuda:0')
    # 2023 7 777
    parser.add_argument('--seed', type=int, default=2024)

    # Experimental setup
    parser.add_argument('--dataset', type=str, default="cifar10", help="cifar10, cifar100, fmnist, svhn")
    parser.add_argument('--algorithm', type=str, default="fedavg")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ratio', type=float, default=1)

    # Test
    parser.add_argument('--test_rounds', type=int, default=1000)
    parser.add_argument('--test_interval', type=int, default=1)

    # Optimizer setup
    parser.add_argument('--optimizer', type=str, default="sgd", help='sgd or other optimizers')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_weight_decay', type=float, default=5e-4)
    parser.add_argument('--adam_weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=bool, default=False)

    # Non-IID setup
    parser.add_argument('--labeled_partition', type=str, default="hybrid",
                        help='client, ratio, hybrid')
    parser.add_argument('--dirichlet', type=float, default=0.3)
    parser.add_argument('--labeled_clients', type=int, default=0)
    parser.add_argument('--part_labeled_clients', type=int, default=5)
    parser.add_argument('--labeled_ratio', type=float, default=0.05)
    parser.add_argument('--split_empty', type=bool, default=True)

    # Warmup setup
    parser.add_argument('--warmup', type=bool, default=False, help='only used in (1,2,7) Cifar100 for for faster convergence')
    parser.add_argument('--warmup_round', type=int, default=20)
    
    args = parser.parse_args()

    return args