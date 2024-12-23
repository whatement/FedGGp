python main.py --dataset fmnist --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
python main.py --dataset cifar10 --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
python main.py --dataset cifar100 --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024 --warmup ture
python main.py --dataset cifar100 --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
python main.py --dataset svhn --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024

python main.py --dataset fmnist --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
python main.py --dataset cifar10 --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
# python main.py --dataset cifar100 --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024 --warmup ture
# python main.py --dataset cifar100 --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024
python main.py --dataset svhn --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 1 --part_labeled_clients 2 --seed 2024


python main.py --dataset fmnist --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset cifar10 --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset cifar100 --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset svhn --algorithm fedggp --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024

python main.py --dataset fmnist --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset cifar10 --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset cifar100 --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024
python main.py --dataset svhn --algorithm twin_sight --gpu cuda:0 --labeled_partition hybrid --labeled_clients 0 --part_labeled_clients 5 --seed 2024