import argparse
import os
def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    # dataset
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_path_cifar100', type=str, default=os.path.join(path_dir, '../data/cifar-100-python'))
    parser.add_argument('--num_data_train', type=int, default=49000)
    parser.add_argument('--gpu', default='1', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--join_ratio', type=float, default=0.5)
    parser.add_argument('--global_learning_rate', type=float, default=0.05)
    parser.add_argument('--local_learning_rate', type=float, default=0.05)
    parser.add_argument('--local_steps', type=int, default=10)
    parser.add_argument('--threshold', default=0.95, type=float,help='pseudo label threshold')
    parser.add_argument('--noise_type', type=str, default='symmetric',help='noise type of each clients')
    parser.add_argument('--noise_rate_list', type=list, default=[],help="noise rate of each clients")
    parser.add_argument('--num_gradual', type=int, default=10, help='T_k')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lamda', type=float, default=0.12)

    #non-iid
    parser.add_argument('--iid', type=int, default=0)
    parser.add_argument('--non_iid_alpha', type=float, default=10)

    #distill
    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--mini_batch_size_distillation', type=int, default=128)
    # parser.add_argument('--mini_batch_size_teaching', type=int, default=20)
    parser.add_argument('--ld', type=float, default=0.5, help='threshold of distillation aggregate')
    parser.add_argument('--eval_step', default=78, type=int , help='number of eval steps to run')

    #other
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args