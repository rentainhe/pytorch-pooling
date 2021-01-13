import argparse
from configs import configs
from utils.train_engine import train_engine

def parse_args():
    parser = argparse.ArgumentParser(description='Pooling Survey Args')
    parser.add_argument('--gpu', type=str, help="gpu choose, eg. '0,1,2,...' ")
    parser.add_argument('--run', type=str, dest='run_mode',choices=['train','test'])
    parser.add_argument('--name', type=str, required=True, help='the name of this training')
    parser.add_argument('--img_size', type=int, default=32, help='Resolution size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help="learning rate decay rate")
    parser.add_argument('--warmup_epoch', type=int, default=1, help='warmup epochs')
    parser.add_argument('--epoch', type=int, default=200, help='total epochs')
    parser.add_argument('--save_epoch', type=int, default=20, help="save model after every 20 epoch")
    parser.add_argument('--eval_every_epoch', action='store_true', default=True, help='evaluate the model every epoch')
    parser.add_argument('--pooling', type=str,
                        choices=[
                            'max',
                            'avg',
                            'mixed',
                            'Lp',
                            'lip',
                            'stochastic',
                            'soft'
                        ], default='max',help='choose one pooling method to use', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args_dict = configs.parse_to_dict(args)
    configs.add_args(args_dict)
    configs.training_init()
    configs.path_init()

    print("Hyper parameters:")
    print(configs)

    if configs.run_mode == 'train':
        train_engine(configs)
