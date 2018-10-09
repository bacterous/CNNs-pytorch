import os
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from torchvision import models

import models
from data.dataset import Echocardiography
from utils.visualize import Visualizer
from config import opt


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    model = getattr(models, opt.models)(opt.num_class)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    train_data = Echocardiography(opt.train_data_root, train=True)
    val_data = Echocardiography(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=True,
                                num_workers=opt.num_workers)

    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=opt.weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_class)
    auc_meter = meter.AUCMeter()

    previous_loss = 100

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        if opt.num_class==2:
            auc_meter.reset()

        for ii, (data, label) in enumerate(train_dataloader):
            input = Variable(data)
            target = Variable(label.long())
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.data)
            confusion_matrix.add(score.data, target.data)
            if opt.num_class==2:
                auc_meter.add(score.data[:, 1], target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                if opt.num_class == 2:
                    vis.plot('auc', auc_meter.value()[0])

        val_cm, val_accuracy = val(model, val_dataloader)[:2]
        if opt.num_class == 2:
            val_auc = val(model, val_dataloader)[2]
            vis.plot('val_auc', val_auc)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}"
            .format(
            epoch=epoch,
            lr=lr,
            loss=loss_meter.value()[0],
            train_cm=str(confusion_matrix.value()),
            val_cm=str(val_cm.value())
        ))
        model.save()
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(opt.num_class)
    auc_meter = meter.AUCMeter()
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.long())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        with t.no_grad():
            score = model(val_input)
            confusion_matrix.add(score.data.squeeze(), label)
            if opt.num_class==2:
                auc_meter.add(score.data[:, 1], label)

    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 1. * np.trace(cm_value) / (cm_value.sum())

    if opt.num_class==2:
        return confusion_matrix, accuracy, auc_meter.value()[0]
    else:
        return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)(opt.num_class).eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    test_data = Echocardiography(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    results = []
    class_prob = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data)
        if opt.use_gpu:
            input = input.cuda()

        with t.no_grad():
            score = model(input)
            probability = t.max(nn.functional.softmax(score, dim=0), 1)[1].data.tolist()
            batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
            results += batch_results
            class_prob += nn.functional.softmax(score, dim=0).data.tolist()

    write_csv(results, class_prob, opt.result_file)
    return results


def write_csv(results, class_prob, file_name):
    results = np.array(results)
    prefix = file_name.split('.')[-2].split('_')[-1]
    results = pd.DataFrame({'path':results[:, 0],
                             '_label':results[:, 1]})

    # \\code\\champ\\mid_data\\results\\T3_chd.csv  chd_0, chd_1
    # \\code\\champ\\mid_data\\result_type.csv      type_0, type_1, ..., type_6
    for i in  range(len(class_prob[0])):
        results[prefix + '_' + str(i)] = np.array(class_prob)[:, i]

    results.to_csv(file_name, index=False)

if __name__ == '__main__':
    import fire
    fire.Fire()
