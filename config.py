""" Config """
import argparse
import os
from omegaconf import OmegaConf

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

class Config(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Config")
        parser.add_argument('--name', required=False, default='cifar10')
        parser.add_argument('--data', help='CIFAR10 (default) / CIFAR100', default='CIFAR10')
        parser.add_argument('--nb_classes', dest='nb_classes', help='the number of classes', default=2, type=int)
        parser.add_argument('--nb_epochs', dest='nb_epochs', help='the number of epochs', default=300, type=int)
        parser.add_argument('--momentum', dest='momentum', help='the momentum', default=0.9, type=float)
        parser.add_argument('--learn_start', dest='learn_start', help='the learning rate at begin', default=0.04,
                            type=float)
        parser.add_argument('--schedule', dest='schedule', help='weight Decrease learning rate', default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
                            type=int,
                            nargs='+')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--batch_size', type=int, default=16, help='default: 8')
        parser.add_argument('--gamma', dest='gamma', help='gamma', default=0.5, type=float)
        parser.add_argument('--train_dir', dest='train_dir', help='training data dir', default="tmp", type=str)
        parser.add_argument('--deepest', dest='deepest', help='Build with only deepest column activated', default=True,
                            type=bool)
        parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=1e-4, type=float)
        parser.add_argument('--load', dest='load', help='Test network with weights file', default=True, type=bool)
        parser.add_argument('--test-all', nargs=1, help='Test all the weights from a folder')
        parser.add_argument('--summary', help='Print a summary of the network and exit', action='store_true')

        parser.add_argument('--init_channels', type=int, default=64,
                            help="doubling each block except the last (default: 64)")
        parser.add_argument('--gdrop_ratio', type=float, default=0.5,
                            help="ratio of global drop path (default: 0.5)")
        parser.add_argument('--p_ldrop', type=float, default=0.15,
                            help="local drop path probability (default: 0.15)")
        parser.add_argument('--cb_dropout', type=float, default=0.2,
                            help="Conv Block dropout probability (default: 0.2)")
        parser.add_argument('--dropout_probs', default="0.0, 0.1, 0.2, 0.3, 0.4",
                            help='dropout probs for each block with comma separated '
                                 '(default: 0.0, 0.1, 0.2, 0.3, 0.4)')
        parser.add_argument('--blocks', type=int, default=5, help='default: 5')
        parser.add_argument('--columns', type=int, default=3, help='default: 3')
        parser.add_argument('--C_L', type=int, default=512, help='size of the Conv kernel at the entrance, default: 1024')
        parser.add_argument('--seed', type=int, default=2022, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aug_lv', type=int, default=0,
                            help='data augmentation level (0~2). 0: no augmentation, '
                                 '1: horizontal mirroring + [-4, 4] translation, '
                                 '2: 1 + cutout.')

        parser.add_argument('--T', type=int, default=10, help='length of the timing serial')
        parser.add_argument('--tau', type=float, default=1.5, help='time constant of the LIF neuron')

        parser.add_argument('--config_file', type=str, default='snn_mlp_1.yaml',
                            help='path to configuration file')
        parser.add_argument('--train', action='store_true',
                            help='train model')
        parser.add_argument('--test', action='store_true',
                            help='test model')
        args = parser.parse_args()

        # EXPERIMENTS
        exp_parser = parser.add_argument_group('Experiment')
        exp_parser.add_argument('--off-drops', action='store_true', default=False,
                                help='turn off all dropout and droppath')
        exp_parser.add_argument('--gap', type=int, default=0, help='0: max-pool (default), '
                                                                   '1: GAP - FC, 2: 1x1conv - GAP')
        exp_parser.add_argument('--init', default='he',
                                help='xavier (default) / he / torch (pytorch default)')
        exp_parser.add_argument('--pad', default='zero', help='zero (default) / reflect')
        exp_parser.add_argument('--doubling', default=False, action='store_true',
                                help='turn on 1x1 conv channel doubling')
        exp_parser.add_argument('--gdrop_type', default='ps-consist',
                                help='ps (per-sample, various gdrop per block) / '
                                     'ps-consist (default; per-sample, consist global drop)')
        exp_parser.add_argument('--dropout_pos', default='CDBR',
                                help='CDBR (default; conv-dropout-BN-relu) / '
                                     'CBRD (conv-BN-relu-dropout) / FD (fractal_block-dropout)')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        #self.path = os.path.join("./runs", self.name)
        #self.check_exists(self.path)

        self.data = self.data.lower().strip()
        self.data_path = './data/'
        self.dropout_probs = [float(p) for p in self.dropout_probs.split(',')]
        self.consist_gdrop = self.gdrop_type == 'ps-consist'
        assert self.gdrop_type in ['ps', 'ps-consist']
        assert len(self.dropout_probs) == self.blocks

        # learning rate decay 4 times.
        # In the case of default epochs 400, lr milestone is = [200, 300, 350, 375].
        left = self.nb_epochs // 2
        self.lr_milestone = [left]
        for i in range(3):
            left //= 2
            self.lr_milestone.append(self.lr_milestone[-1] + left)

        if self.off_drops:
            print("\n!!! Dropout and droppath are off !!!\n")
            for i in range(self.blocks):
                self.dropout_probs[i] = 0.
            self.p_ldrop = 0.
            self.gdrop_ratio = 0.

class TestConfig(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("Config")
        parser.add_argument('--name', required=False, default='cifar10')
        parser.add_argument('--data', help='CIFAR10 (default) / CIFAR100', default='CIFAR10')
        parser.add_argument('--batch_size', type=int, default=64, help='default: 64')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--init_channels', type=int, default=64,
                            help="doubling each block except the last (default: 64)")
        parser.add_argument('--blocks', type=int, default=5, help='default: 5')
        parser.add_argument('--columns', type=int, default=3, help='default: 3')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')

        # EXPERIMENTS
        exp_parser = parser.add_argument_group('Experiment')
        exp_parser.add_argument('--gap', type=int, default=0, help='0: max-pool (default), '
                                '1: GAP - FC, 2: 1x1conv - GAP')
        exp_parser.add_argument('--pad', default='zero', help='zero (default) / reflect')
        exp_parser.add_argument('--doubling', default=False, action='store_true',
                                help='turn on 1x1 conv channel doubling')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data = self.data.lower().strip()
        self.data_path = './data/'