import os
import math
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prettytable import PrettyTable

from SMPNet import SMPNet
from data import DatasetFromFolder
from NegPearsonLoss import Neg_Pearson

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HR estimation')
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testbatchSize', type=int, default=32, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--w_hr', default=0.2, help='weight of L1Loss_hr')
parser.add_argument('--w_ppg', default=10, help='weight of negPearsonLoss_ppg')
parser.add_argument('--w_rr', default=2, help='weight of L1Loss_rr')

parser.add_argument('--save_folder', default='model/', help='Location to save checkpoint models')
parser.add_argument('--path_source', type=str, default='/home/som/8T/kz/DataProcessing/MMVS/')
parser.add_argument('--traindata_path', type=str, default='trainlabels-0118.npy')
parser.add_argument('--testdata_path', type=str, default='testlabels-0118.npy')

local_rank = int(os.environ['LOCAL_RANK'])
opt = parser.parse_args()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://')

cudnn.benchmark = True
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(opt.seed)


def model_train(epoch):
    epoch_loss = 0
    model.train()
    loss_task = []
    bar = tqdm(training_data_loader)
    for batch in bar:
        imgarr_v,imgarr_i, label_hr, label_ppg, label_rr = batch[0], batch[1], batch[2], batch[3], batch[4]
        if cuda:
            imgarr_v = Variable(imgarr_v).cuda(local_rank)
            imgarr_i = Variable(imgarr_i).cuda(local_rank)
            label_hr = Variable(label_hr).cuda(local_rank)
            label_ppg = Variable(label_ppg).cuda(local_rank)
            label_rr = Variable(label_rr).cuda(local_rank)
        pred_hr, pred_ppg, pred_rr = model(imgarr_v,imgarr_i)
        pred_hr = pred_hr.squeeze(-1).type_as(label_hr)
        pred_rr = pred_rr.squeeze(-1).type_as(label_rr)
        loss_hr = criterion_hr(pred_hr, label_hr)
        loss_ppg = criterion_ppg(pred_ppg, label_ppg)
        loss_rr = criterion_hr(pred_rr, label_rr)
        loss_task.append([loss_hr, loss_ppg, loss_rr])
        loss = w_hr * loss_hr + w_ppg * loss_ppg + w_rr * loss_rr
        epoch_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_description("Epoch {}/{}".format(epoch, opt.nEpochs))
    loss_task = torch.tensor(loss_task)
    print("===> Epoch {} Complete: Loss_all={:.3f},L1Loss_hr={:.3f},NegPearsonLoss_ppg={:.3f},L1Loss_rr={:.3f}"
          .format(epoch, epoch_loss / len(training_data_loader), torch.mean(loss_task[:, 0]),
                  torch.mean(loss_task[:, 1]), torch.mean(loss_task[:, 2])))
    loss_save[epoch, :3] = [torch.mean(loss_task[:, 0]), torch.mean(loss_task[:, 1]), torch.mean(loss_task[:, 2])]


def calculate_indexes_hr(y_pred, y_true):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mer = torch.sum(torch.abs((y_true - y_pred))) / len(y_true) / torch.mean(y_true) * 100
    r, p = pearsonr(y_pred, y_true)
    return [round(mae, 2), round(rmse, 2), round(mer.item(), 2), round(r, 2)]


def model_eval(epoch):
    model.eval()
    preds_hr = torch.tensor([])
    preds_ppg = torch.tensor([])
    preds_rr = torch.tensor([])
    labels_hr = torch.tensor([])
    labels_ppg = torch.tensor([])
    labels_rr = torch.tensor([])
    for iteration, batch in enumerate(testing_data_loader, 1):
        imgarr_v,imgarr_i, label_hr, label_ppg, label_rr = batch[0], batch[1], batch[2], batch[3], batch[4]
        labels_hr = torch.cat((labels_hr, label_hr), 0)
        labels_ppg = torch.cat((labels_ppg, label_ppg), 0)
        labels_rr = torch.cat((labels_rr, label_rr), 0)
        with torch.no_grad():
            imgarr_v = Variable(imgarr_v).cuda(local_rank)
            imgarr_i = Variable(imgarr_i).cuda(local_rank)
            pred_hr, pred_ppg, pred_rr = model(imgarr_v,imgarr_i)
        preds_hr = torch.cat((preds_hr, pred_hr[:, 0].cpu()), 0)
        preds_ppg = torch.cat((preds_ppg, pred_ppg.cpu()), 0)
        preds_rr = torch.cat((preds_rr, pred_rr[:, 0].cpu()), 0)
    neg_p = criterion_ppg(preds_ppg, labels_ppg)
    p_ppg = round((1 - neg_p.item()), 2)
    index_hr = calculate_indexes_hr(preds_hr, labels_hr)
    index_rr = calculate_indexes_hr(preds_rr, labels_rr)
    loss_save[epoch, 3:] = [index_hr[0], index_hr[1], index_hr[2], index_hr[3], index_rr[0], index_rr[1], index_rr[2],
                            index_rr[3], round(neg_p.item(), 2)]
    table = PrettyTable(["Signal", "MAE", "RMSE", "MER", "Pearson_r"])
    table.add_row(["H R", index_hr[0], index_hr[1], index_hr[2], index_hr[3]])
    table.add_row(["R R", index_rr[0], index_rr[1], index_rr[2], index_rr[3]])
    table.add_row(["PPG", " ", " ", " ", p_ppg])
    print(table)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint():
    model_out_path = opt.save_folder + "epoch_1213_rr05.pth"
    torch.save(model.module, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    print('===> Loading datasets')
    path_data_train = os.path.join(opt.path_source, opt.traindata_path)
    train_set = DatasetFromFolder(path_data_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=False, sampler=train_sampler)

    path_data_test = os.path.join(opt.path_source, opt.testdata_path)
    test_set = DatasetFromFolder(path_data_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchSize,
                                     shuffle=False, sampler=test_sampler)

    print('===> Building model ')
    model = SMPNet().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    criterion_hr = nn.L1Loss().cuda(local_rank)
    criterion_ppg = Neg_Pearson().cuda(local_rank)

    w_hr = opt.w_hr
    w_ppg = opt.w_ppg
    w_rr = opt.w_rr
    test_loss_best = 100.0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    loss_save = np.zeros((opt.nEpochs, 12))

    print('===> Starting model ')
    for epoch in range(opt.nEpochs):
        model_train(epoch)
        if epoch % 2 == 0:
            model_eval(epoch)
            test_loss_temp = w_hr * loss_save[epoch][3] + w_rr * loss_save[epoch][7] + w_ppg * loss_save[epoch][-1]
            if test_loss_temp < test_loss_best:
                test_loss_best = test_loss_temp
                checkpoint()

        if epoch!=0 and epoch % (opt.nEpochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    name_results = 'results12-13-rr05.npy'
    np.save(name_results, loss_save)
