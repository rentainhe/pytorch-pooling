import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from models.get_network import get_network
import argparse
from models.net.vgg import vgg16_bn


def count_parameters(net):
    params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    print("Params: %f M" % (params / 1000000))


def parse_args():
    parser = argparse.ArgumentParser(description='MAC_Classification Args')
    parser.add_argument('--pooling', type=str,
                        choices=[
                            'max',
                            'avg',
                            'mixed',
                            'Lp',
                            'lip',
                            'stochastic',
                            'soft'
                        ], default='max', help='choose one pooling method to use', required=True)
    parser.add_argument('--gpu', type=int, default=0, help="choose a gpu for testing")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    if device == 'cpu':
        net = vgg16_bn(args)
        count_parameters(net)
    else:
        with torch.cuda.device(args.gpu):
            net = vgg16_bn(args)
            macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
