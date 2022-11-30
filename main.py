import sys
import os
import io
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from spikingjelly.clock_driven import neuron, layer, functional, surrogate
import time
import scipy.io as sio

from omegaconf import OmegaConf

import model
from config import Config
from spikingjelly.clock_driven import encoding
import utils
from model import LIF_FractalNet  # ,TEST_MODEL
import wandb

args = Config()

"""
超参数配置
"""

"""
parser = argparse.ArgumentParser(description='Weight Decay Experiments')


#parser.add_argument('--config_file', type=str, default='snn_mlp_1.yaml', help='path to configuration file')
parser.add_argument('--train', action='store_true',
                    help='train model')
parser.add_argument('--test', action='store_true',
                    help='test model')
args = parser.parse_args()

config = Config()
conf = OmegaConf.load(args.config_file)

torch.manual_seed(conf['pytorch_seed'])
np.random.seed(conf['pytorch_seed'])
experiment_name = conf['experiment_name']

# %% training parameters
hyperparam_conf = conf['hyperparameters']
length = hyperparam_conf['length']
batch_size = hyperparam_conf['batch_size']
synapse_type = hyperparam_conf['synapse_type']
epoch = hyperparam_conf['epoch']
tau_m = hyperparam_conf['tau_m']
tau_s = hyperparam_conf['tau_s']
tau = hyperparam_conf['tau']



membrane_filter = hyperparam_conf['membrane_filter']

train_bias = hyperparam_conf['train_bias']
train_coefficients = hyperparam_conf['train_coefficients']

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

# logger
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
logger.info("Run options = {}".format(sys.argv))
config.print_params(logger.info)

# copy scripts
utils.copy_scripts("*.py", config.path)
"""


class network(nn.Module):
    def __init__(self, args, data_shape, batch_size):
        super(network, self).__init__()
        length = data_shape[0]
        self.dropout = [0.1]
        # self.conv = [(length, 512), (512, 256), (256, 128)]
        # self.c, self.b = 1, 1
        # num_columns = self.c
        p_ldrop = args.p_ldrop
        cb_dropout=args.cb_dropout
        dropout_probs = args.dropout_probs
        gdrop_ratio = args.gdrop_ratio
        tau = args.tau
        T = args.T
        num_columns = args.columns
        num_blocks = args.blocks
        init_channels = args.init_channels
        channel = data_shape[1]

        """
            self.lif_fractal_net = LIF_FractalNet(
            c=self.c, b=self.b, conv=self.conv,
            drop_path=0.15, dropout=self.dropout,
            deepest=deepest, length=length,
                batch_size=batch_size,
                train_coefficients=train_coefficients,
                train_bias=train_bias, membrane_filter=membrane_filter, tau_m=tau_m, tau_s=tau_s, class_num=class_num)
        # self.linear = nn.Linear(neuron_nb, 1)
        """
        C_L = args.C_L
        self.T = T

        self.conv0 = nn.Conv1d(length, C_L, kernel_size=3, padding="same", bias=False)

        self.encoder = encoding.PoissonEncoder()
        self.lif_fractal_net = LIF_FractalNet(T, data_shape, num_columns, init_channels, p_ldrop, dropout_probs,
                                              batch_size, cb_dropout, C_L, tau,
                                              gdrop_ratio, gap=0, init='xavier', pad_type='zero', doubling=False,
                                              consist_gdrop=True, dropout_pos='CDBR')

        # self.testsnn=TEST_MODEL(tau)

    def forward(self, x):
        """
        x = self.conv0(x)
        x = self.encoder(x)
        """
        x_conv0 = self.conv0(x).swapaxes(2,1)
        x_encoder = self.encoder(x_conv0).unsqueeze(0).repeat(self.T, 1, 1, 1)

        out = self.lif_fractal_net(x_encoder)

        # x=self.testsnn(x)

        # x = self.softmax(x)
        return out


def train_network(epoch_no, test_sample_no, data_loader, model, criterion, optimizer, lr_scheduler):
    with torch.autograd.set_detect_anomaly(True):
        print("Training network")
        model.train(mode=True)
        begin = time.time()
        train_correct_sum = 0
        train_sum = 0
        losses=0.0

        train_accs = []

        for j, (input, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(non_blocking=True)

            # input = input.repeat(length, 1, 1).permute(1, 2, 0)

            # a = torch.max(input)
            input = input.float()
            """
            print("input")
            print(input)            
            """
            optimizer.zero_grad()
            input.requires_grad_()
            output = model(input)
            """
            for t in range(args.T):

                if t == 0:
                    out_spikes_counter = output[t, ].float()
                else:
                    out_spikes_counter = out_spikes_counter+output[t, ].float()

            print("\n theta\n")
            print(theta)
            out_spikes_counter_frequency = out_spikes_counter / args.T
            print("\n out_spikes_counter\n")
            print(out_spikes_counter)
            print("\n out_spikes_counter_frequency\n")
            print(out_spikes_counter_frequency)
            print("\n out_spikes_counter_frequency_counter\n")
            out_spikes_counter_frequency_counter, indice = out_spikes_counter_frequency.max(1)
            #out_spikes_counter_frequency_counter = torch.argmax(out_spikes_counter_frequency, 1)
            print(out_spikes_counter_frequency_counter)            

            print("\n out_spikes_counter\n")

            """

            out_spikes_counter_frequency = output.mean(0)
            target_one_hot = nn.functional.one_hot(target.to(torch.int64), args.nb_classes).float()
            """

            print(output)

            print("\n out_spikes_frequency\n")
            print(out_spikes_counter_frequency)
            print("\n target_one_hot\n")
            print(target_one_hot)
            """

            loss = criterion(out_spikes_counter_frequency, target_one_hot.float())
            loss.backward()
            # loss.backward(retain_graph=True)
            # losses = losses + float(loss.item())
            losses = float(loss.item())  # 记录一个batch的loss
            optimizer.step()
            functional.reset_net(model)

            train_correct_sum = train_correct_sum + (torch.argmax(out_spikes_counter_frequency, 1) == target).sum().item()
            train_sum = train_sum + target.numel()
            batch_size = args.batch_size
            train_batch_accuracy = (torch.argmax(out_spikes_counter_frequency, 1) == target).float().mean().item()
            be_time = time.time() - begin
            begin = time.time()
            train_accuracy = train_correct_sum / train_sum
            train_accs.append(train_accuracy)

            print("train:epoch {} [{}/{}] loss is {:.3f},acc is {:.3f},time is {:.3f}".format(epoch_no,
                                                                                              j * batch_size,
                                                                                              len(
                                                                                                  data_loader) * batch_size,
                                                                                              float(
                                                                                                  losses),
                                                                                              float(
                                                                                                  train_accuracy),
                                                                                              float(
                                                                                                  be_time)))
            # wandb.log({"loss": loss})

            # Optional
            # wandb.watch(model)
            wandb.log({
                "Train Accuracy": 100. * train_accuracy,
                "Train Loss": losses,
            })

        print("\n out_spikes_frequency\n")
        print(out_spikes_counter_frequency)
        print("\n target_one_hot\n")
        print(target_one_hot)

        return model, losses, train_accs


def test_network(epoch_no, test_sample_no, data_loader, model, criterion):
    test_losses = 0.0
    model.eval()
    begin = time.time()
    test_correct_sum = 0
    test_sum = 0
    batch_size = args.batch_size
    test_accs = []

    with torch.no_grad():
        for j, (input, target) in enumerate(data_loader):
            if torch.cuda.is_available():
                input = input.cuda().float()
                target = target.cuda(non_blocking=True)
            if torch.cuda:
                torch.cuda.synchronize()
            test_output = model(input)
            test_out_spikes_counter_frequency = test_output.mean(0)

            """
            for t in range(args.T):
                if t == 0:
                    out_spikes_counter = output[t, ].float()
                else:
                    out_spikes_counter = out_spikes_counter+output[t, ].float()
            out_spikes_counter_frequency = out_spikes_counter / args.T

            print("\n theta\n")
            print(theta)
            print("\n frequency\n")
            print(out_spikes_counter_frequency)
            print("\n counter\n")
            print(out_spikes_counter_frequency_counter)
            out_spikes_counter_frequency_counter, indice = out_spikes_counter_frequency.max(1)
            """
            print("\n test_out_spikes_frequency\n")
            print(test_out_spikes_counter_frequency)
            print("\n test_target\n")
            target_one_hot = torch.squeeze(nn.functional.one_hot(target.to(torch.int64), args.nb_classes).float())
            print(target_one_hot)
            loss = criterion(test_out_spikes_counter_frequency, target_one_hot.float())
            test_losses = test_losses + float(loss.item())

            test_sum = target.numel()
            print("test_out")
            print(torch.argmax(test_out_spikes_counter_frequency, 1).reshape(test_sum, 1))
            test_correct_sum = (torch.argmax(test_out_spikes_counter_frequency, 1).reshape(test_sum,1) == target.reshape(test_sum,1)).float().sum()
            """
            _, idx = torch.max(output, dim=1)

            eval_image_number = eval_image_number + len(target)

            right = len(torch.where(idx == target)[0])
            acc = acc + right
            acc = acc / eval_image_number
                """
            functional.reset_net(model)
            be_time = time.time() - begin
            begin = time.time()

        test_accuracy = (test_correct_sum / test_sum).float()
        print("test: loss is {:.3f},acc is {:.3f},time is {:.3f}".format(
            float(test_losses / len(data_loader)),
            float(test_accuracy),
            float(be_time)))

        # Optional
        # wandb.watch(model)

        wandb.log({
            "Test Accuracy": 100. * test_accuracy,
            "Test Loss": test_losses / len(data_loader)
        })
    return test_losses, test_accuracy


def index_shift(data, count, axis=1):
    num = data.shape[axis]
    temp = np.array(range(count))
    index = np.array(range(num))
    index = index + count
    index[-count:] = temp
    if axis == 0:
        return data[index, ...]
    else:
        return data[:, index, ...]

def trying_on_dreamer(args, batch_size, label):
    # args.dropout_probs = [0.5]

    # # label_selection2 = 'ScoreValence'
    # classfication = 2  # 2 or 4
    # mode = 'periodogram'  # 'periodogram', 'stft'
    # repeat = 0
    # scale = 0.1
    # if repeat == 0:
    #     scale = 0.1
    # test_size = 0.3
    # dataSetseed = 200
    # X_train = np.load(file='class%d\Xtrain_ts%dre%ds%d_%s_modelseed%d.npy' % (
    #     classfication, test_size * 100, repeat, scale * 100, label_selection, dataSetseed))
    # y_train = np.load(file='class%d\ytrain_ts%dre%ds%d_%s_modelseed%d.npy' % (
    #     classfication, test_size * 100, repeat, scale * 100, label_selection, dataSetseed))
    # X_test = np.load(file='class%d\Xtest_ts%dre%ds%d_%s_modelseed%d.npy' % (
    #     classfication, test_size * 100, repeat, scale * 100, label_selection, dataSetseed))
    # y_test = np.load(file='class%d\ytest_ts%dre%ds%d_%s_modelseed%d.npy' % (
    #     classfication, test_size * 100, repeat, scale * 100, label_selection, dataSetseed))
    #
    # # load training dataset
    # trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # # load test dataset
    # testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    #
    # train_data = MNISTDataset(trainset, max_rate=1, length=length, flatten=True) # （64，32）flatten 后变成了2048
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_data = MNISTDataset(testset, max_rate=1, length=length, flatten=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()

    '''dreamer 时域'''
    X = np.load('.\data\dreamer_raw.npy').reshape(23 * 18, 7680, -1)
    Y = np.load('./data/dreamer_label_%s.npy' % label).reshape(23 * 18)

    """
    # X = X.reshape(-1, X.shape[2], X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=1/23, random_state=100)

    X_train = X_train.reshape(-1, 7680, 14)
    X_test = X_test.reshape(-1, 7680, 14)

    # load training dataset
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # load test dataset
    testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    """

    # train_loader, test_loader = load_data(batch_size, batch_size)
    # writer = SummaryWriter(args.train_dir)

    data_shape = []
    data_shape.append(X.shape[1])
    data_shape.append(X.shape[2])
    data_shape.append(args.nb_classes)

    """
    model = network(args, data_shape, batch_size).cuda()
    print(model)
    """

    for j in range(23):
        wandb.init(config=args, project="FSNN_DREAMER_20221130", group="DREAMER_LOSO_LABEL_"+label,
                   name="FB_WITH_T=" + str(args.T) + "_BATCH_SIZE=" + str(batch_size) + "__C_L=" + str(
                       args.C_L) + "_No." + str(j), entity="g-beta", reinit=True)
        X_train = np.append(X[:18 * j][:][:], X[18 * j + 18:][:][:], 0)
        y_train = np.append(Y[:18 * j][:], Y[18 * j + 18:][:])
        X_test = X[18 * j:18 * j + 18][:][:]
        y_test = Y[18 * j:18 * j + 18][:]
        # load training dataset
        trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        # load test dataset
        testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(testset, batch_size=18, shuffle=True, drop_last=False)
        model = network(args, data_shape, batch_size)
        if (j == 0):
            print(model)
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learn_start,
                                    momentum=args.momentum, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
        for i in range(args.nb_epochs):
            model, train_losses, train_accs = train_network(i, j, train_loader, model, criterion, optimizer,
                                                            lr_scheduler)
            test_losses, test_acc = test_network(i, j, test_loader, model, criterion)
            print(test_acc)

        wandb.log({
            "Epoch Final Train Accuracy": 100. * train_accs[-1],
            "Avg Train Loss": train_losses / len(train_loader),
            "Test Accuracy": 100. * test_acc,
            "Avg Test Loss": test_losses / len(test_loader)
        }, j)


# Optional
# wandb.watch(model, log="all")

def trying_on_deap(time_step, args, batch_size, label):
    criterion = nn.MSELoss()

    '''deap 时域'''
    X = np.load('E:/DATASET/DEAP_EEG.npy')
    Y = np.load('E:/DATASET/DEAP_Binary_%s.npy' % label)

    """
    # X = X.reshape(-1, X.shape[2], X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=1/23, random_state=100)

    X_train = X_train.reshape(-1, 7680, 14)
    X_test = X_test.reshape(-1, 7680, 14)

    # load training dataset
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # load test dataset
    testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    """

    # train_loader, test_loader = load_data(batch_size, batch_size)
    # writer = SummaryWriter(args.train_dir)

    data_shape = []
    data_shape.append(X.shape[1])
    data_shape.append(X.shape[2])
    data_shape.append(args.nb_classes)

    """
    model = network(args, data_shape, batch_size).cuda()
    print(model)
    """

    for j in range(32):
        wandb.init(config=args, project="FSNN_DEAP_20221130", group="DEAP_LOSO_LABEL_"+label,
                   name="FSNN_WITH_T=" + str(args.T) +"_C="+str(args.columns)+"_B="+str(args.blocks)+"_BATCH_SIZE=" + str(batch_size) + "__C_L=" + str(
                       args.C_L) + "_No." + str(j), entity="g-beta", reinit=True)
        X_train = np.append(X[:40 * j][:][:], X[40 * j + 40:][:][:], 0)
        y_train = np.append(Y[:40 * j][:], Y[40 * j + 40:][:])
        X_test = X[40 * j:40 * j + 40][:][:]
        y_test = Y[40 * j:40 * j + 40][:]
        # load training dataset
        trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        # load test dataset
        testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(testset, batch_size=40, shuffle=True, drop_last=False)
        model = network(args, data_shape, batch_size)
        if (j == 0):
            print(model)
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learn_start,
                                    momentum=args.momentum, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
        for i in range(args.nb_epochs):
            model, train_losses, train_accs = train_network(i, j, train_loader, model, criterion, optimizer,
                                                            lr_scheduler)
            test_losses, test_acc = test_network(i, j, test_loader, model, criterion)
            print(test_acc)

        wandb.log({
            "Epoch Final Train Accuracy": 100. * train_accs[-1],
            "Avg Train Loss": train_losses / len(train_loader),
            "Test Accuracy": 100. * test_acc,
            "Avg Test Loss": test_losses / len(test_loader)
        }, j)

def trying_on_seed(time_step, args, batch_size):
    criterion = nn.MSELoss()

    '''deap 时域'''
    X = np.load('E:/DATASET/seed_data.npy').reshape(15 * 45, 12000, -1)
    Y = np.load('E:/DATASET/seed_label_012.npy')
    args.nb_classes = 3

    """
    # X = X.reshape(-1, X.shape[2], X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=1/23, random_state=100)

    X_train = X_train.reshape(-1, 7680, 14)
    X_test = X_test.reshape(-1, 7680, 14)

    # load training dataset
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    # load test dataset
    testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    """

    # train_loader, test_loader = load_data(batch_size, batch_size)
    # writer = SummaryWriter(args.train_dir)

    data_shape = []
    data_shape.append(X.shape[1])
    data_shape.append(X.shape[2])
    data_shape.append(args.nb_classes)

    """
    model = network(args, data_shape, batch_size).cuda()
    print(model)
    """

    for j in range(15):
        wandb.init(config=args, project="FSNN_SEED_20221130", group="SEED_LOSO",
                   name="FSNN_WITH_T=" + str(args.T) +"_C="+str(args.columns)+"_B="+str(args.blocks)+"_BATCH_SIZE=" + str(batch_size) + "__C_L=" + str(
                       args.C_L) + "_No." + str(j), entity="g-beta", reinit=True)
        X_train = np.append(X[:45 * j][:][:], X[45 * j + 45:][:][:], 0)
        y_train = np.append(Y[:45 * j][:], Y[45 * j + 45:][:])
        X_test = X[45 * j:45 * j + 45][:][:]
        y_test = Y[45 * j:45 * j + 45][:]
        # load training dataset
        trainset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        # load test dataset
        testset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader = DataLoader(testset, batch_size=45, shuffle=True, drop_last=False)
        model = network(args, data_shape, batch_size)
        if (j == 0):
            print(model)
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
        optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learn_start,
                                    momentum=args.momentum, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)
        for i in range(args.nb_epochs):
            model, train_losses, train_accs = train_network(i, j, train_loader, model, criterion, optimizer,
                                                            lr_scheduler)
            test_losses, test_acc = test_network(i, j, test_loader, model, criterion)
            print(test_acc)

        wandb.log({
            "Epoch Final Train Accuracy": 100. * train_accs[-1],
            "Avg Train Loss": train_losses / len(train_loader),
            "Test Accuracy": 100. * test_acc,
            "Avg Test Loss": test_losses / len(test_loader)
        }, j)

def main():
    label = "ScoreDominance"  # ScoreArousal; ScoreValence; ScoreDominance
    batch_size = args.batch_size
    # structure&dropout_probs
    args.blocks = 2
    args.columns = 4
    args.dropout_probs = [0.2, 0.2]

    #trying_on_dreamer(args, batch_size, label)

    """
    label="Score_A"
    time_step=5
    trying_on_deap(time_step, args, batch_size, label)
    """

    time_step=5
    trying_on_seed(time_step, args, batch_size)


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    main()
