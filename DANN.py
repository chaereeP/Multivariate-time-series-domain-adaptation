import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import h5py
import re
import argparse
import torch.jit as jit
from torch.nn import Parameter
import numbers
from AE_model import Build_DANN, fix_randomness
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import itertools
import random
import torch.backends.cudnn as cudnn
deterministic = False # 모델 구현 및 코드 배포시에만 True, 학습에서는 False
cudnn.benchmark = True

if deterministic:
	cudnn.deterministic = True
	cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train = True
load = True
backbone = 'LSTM' # LSTM (R-DANN) or VRNN(VRADA) or CNN(CoDATS)

################## Load data set##################
num_data = 5
from preprocess import AE_preprocess
preprocess = AE_preprocess()
train_dataset1, test_dataset1, train_dataset2, test_dataset2 = preprocess.data_preprocess() ### LOAD DATASET


def source_only(model, backbone, num_data, x_sim, kinematics):
    train_kine_loss= 0
    loss_shape = nn.MSELoss().to(device) #input, target # for SHAPE
    loss_classifier = nn.BCEWithLogitsLoss().to(device) #input, target # for DOMAIN, TASK CLASSIFICATION

    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(),'lr': 0.0004},\
                                 {'params': model.kinematics.parameters()}], lr=0.001,weight_decay = 1e-6)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                         lr_lambda=lambda epoch: 0.95 ** epoch,
    #                                         last_epoch=-1,
    #                                         verbose=False)
    for i in range(num_data):
        optimizer.zero_grad()

        if backbone == 'VRNN':
            _,_,_,_, z_sim = model.feature_extractor(x_sim[i][..., :5])
            # z_sim = model.feature_extractor.sample(x_sim[i][..., :5])
            z_sim = z_sim.reshape(1,-1,5)
        elif backbone == 'LSTM':
            z_sim = model.feature_extractor(x_sim[i][..., :5])

        elif backbone == 'CNN':
            z_sim = model.feature_extractor(x_sim[i][..., :5])
        if torch.any(torch.isnan(z_sim)):
            print(z_sim)
            raise
        latent = torch.cat([z_sim, x_sim[i][..., 5:]], dim=2)
        pred_shape = model.kinematics(latent)

        if kinematics[i].shape != pred_shape.shape:
            raise
        kine_loss = loss_shape(pred_shape, kinematics[i])  # input, target

        loss = kine_loss
        loss.backward()
        optimizer.step()
        train_kine_loss += kine_loss.mean().item()
    # scheduler.step()
    return model , (train_kine_loss/num_data)

def dann(model, backbone, num_data,x_sim, x_real, kinematics, epoch , n_epoch):
    train_domain_loss, train_kine_loss, train_domain_diff= 0,0,0

    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(),'lr': 0.0004},{'params': model.domain_classifier.parameters()},\
                                 {'params': model.kinematics.parameters()}], lr=0.001, weight_decay = 1e-6)

    loss_classifier = nn.BCEWithLogitsLoss().to(device) #input, target # for DOMAIN, TASK CLASSIFICATION
    loss_shape = nn.MSELoss().to(device) #input, target # for SHAPE
    for j in range(5):
        for i in range(num_data):
            optimizer.zero_grad()
            p = float( i + epoch * 13) / ( n_epoch * 13)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if backbone == 'VRNN':
                kld_loss1, nll_loss1, enc1, dec1, z_sim = model.feature_extractor(x_sim[i][..., :5])
                kld_loss2, nll_loss2, enc2, dec2, z_real = model.feature_extractor(x_real[i][..., :5])
                L_r = kld_loss1 + nll_loss1 + kld_loss2 + nll_loss2

            elif backbone == 'LSTM':
                L_r = torch.tensor([0]).to(device)
                z_sim = model.feature_extractor(x_sim[i][..., :5])
                z_real = model.feature_extractor(x_real[i][..., :5])

            elif backbone == 'CNN':
                L_r = torch.tensor([0]).to(device)
                z_sim = model.feature_extractor(x_sim[i][..., :5])
                z_real = model.feature_extractor(x_real[i][..., :5])
            if torch.any(torch.isnan(z_sim)):
                print(z_sim)
                raise
            if torch.any(torch.isnan(z_real)):
                print(z_real)
                raise
            latent = torch.cat([z_sim, x_sim[i][..., 5:]], dim=2)
            pred_shape = model.kinematics(latent)

            if kinematics[i].shape != pred_shape.shape:
                raise
            kine_loss = loss_shape(pred_shape, kinematics[i])  # input, target

            pred_domain_s = model.domain_classifier(z_sim, alpha) #batch 2
            pred_domain_t = model.domain_classifier(z_real, alpha) #batch 2

            domain_loss = (loss_classifier(pred_domain_s.to(torch.float32), torch.tensor([1]).long().to(torch.float32).to(device)) +\
                           loss_classifier(pred_domain_t.to(torch.float32), torch.tensor([0]).long().to(torch.float32).to(device)))
            # [s, t]=[1,0] Source = 1, Target = 0 so trains as [s, t]=[0,1] source = 0, target = 1
            loss = kine_loss + L_r/2 + domain_loss/2
            loss.backward()
            optimizer.step()
            train_domain_diff += torch.mean((z_sim - z_real)**2)
            train_domain_loss += domain_loss.mean().item()/2
            train_kine_loss += kine_loss.mean().item()

    return model , (train_kine_loss/num_data/5, train_domain_loss/num_data/5, train_domain_diff/num_data/5)


def domain_train(model, backbone, num_data,x_sim, x_real, epoch , n_epoch):
    train_domain_loss, train_domain_diff = 0, 0

    optimizer = torch.optim.Adam([{'params': model.feature_extractor.parameters(), 'lr': 0.0004},
                                  {'params': model.domain_classifier.parameters()}], lr=0.001, weight_decay=1e-6)

    loss_classifier = nn.BCEWithLogitsLoss().to(device)  # input, target # for DOMAIN, TASK CLASSIFICATION
    loss_shape = nn.MSELoss().to(device)  # input, target # for SHAPE
    for j in range(5):
        for i in range(num_data):
            optimizer.zero_grad()
            p = float(i + epoch * 13) / (n_epoch * 13)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if backbone == 'VRNN':
                kld_loss1, nll_loss1, enc1, dec1, z_sim = model.feature_extractor(x_sim[i][..., :5])
                kld_loss2, nll_loss2, enc2, dec2, z_real = model.feature_extractor(x_real[i][..., :5])
                L_r = kld_loss1 + nll_loss1 + kld_loss2 + nll_loss2

            elif backbone == 'LSTM':
                L_r = torch.tensor([0]).to(device)
                z_sim = model.feature_extractor(x_sim[i][..., :5])
                z_real = model.feature_extractor(x_real[i][..., :5])

            elif backbone == 'CNN':
                L_r = torch.tensor([0]).to(device)
                z_sim = model.feature_extractor(x_sim[i][..., :5])
                z_real = model.feature_extractor(x_real[i][..., :5])
            if torch.any(torch.isnan(z_sim)):
                print(z_sim)
                raise
            if torch.any(torch.isnan(z_real)):
                print(z_real)
                raise

            pred_domain_s = model.domain_classifier(z_sim, alpha)  # batch 2
            pred_domain_t = model.domain_classifier(z_real, alpha)  # batch 2

            domain_loss = (loss_classifier(pred_domain_s.to(torch.float32),
                                           torch.tensor([1]).long().to(torch.float32).to(device)) + \
                           loss_classifier(pred_domain_t.to(torch.float32),
                                           torch.tensor([0]).long().to(torch.float32).to(device)))
            # [s, t]=[1,0] Source = 1, Target = 0 so trains as [s, t]=[0,1] source = 0, target = 1
            loss =  L_r / 2 + domain_loss / 2
            loss.backward()
            optimizer.step()
            train_domain_diff += torch.mean((z_sim - z_real) ** 2)
            train_domain_loss += domain_loss.mean().item() / 2

    return model, (train_domain_loss / num_data / 5,
                   train_domain_diff / num_data / 5)
def run(model, backbone, model_path, writer):
    [x_real, x_sim,train_kine1, _] = train_dataset1
    [train_input,_,train_kine2] = train_dataset2

    [test_real, test_input1, test_kine1, _] = test_dataset1
    [test_input2, _, test_kine2] = test_dataset2

    ## 학습하기
    loss_shape = nn.MSELoss().to(device) #input, target # for SHAPE
    loss_classifier = nn.BCEWithLogitsLoss().to(device) #input, target # for DOMAIN, TASK CLASSIFICATION

    ts = datetime.now().replace(microsecond=0)
    start_time = ts

    training_end_num = 0
    min_kine_loss = 1000
    n_epoch = 90000
    for epoch in range(90000):
        model.train()
        # Train Generator and Classifier
        if epoch < 100:
            model, (train_domain_loss, train_domain_diff) = domain_train(model, backbone,len(x_sim), x_sim, x_real,
                                                                                                      epoch, n_epoch)
            continue

        model, (train_kine1_loss, train_domain_loss, train_domain_diff) = dann(model, backbone, len(x_sim), x_sim, x_real,train_kine1, epoch , n_epoch)

        model, (train_kine2_loss) = source_only(model, backbone, len(train_input), train_input, train_kine2)
        # Train Domain Discriminator
        train_loss = (train_kine1_loss)+(train_kine2_loss)+train_domain_loss
        model.eval()
        with torch.no_grad():
            p = float(epoch * 13) / (n_epoch * 13)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            if backbone == 'VRNN':
                kld_loss, nll_loss, enc, dec, z_sim1 = model.feature_extractor(test_input1[..., :5])
                kld_loss, nll_loss, enc, dec, z_sim2 = model.feature_extractor(test_input2[..., :5])
                kld_loss, nll_loss, enc, dec, z_real = model.feature_extractor(test_real[..., :5])

            elif backbone == 'LSTM':
                z_sim1 = model.feature_extractor(test_input1[..., :5])
                z_sim2 = model.feature_extractor(test_input2[..., :5])
                z_real = model.feature_extractor(test_real[..., :5])

            elif backbone == 'CNN':
                z_sim1 = model.feature_extractor(test_input1[..., :5])
                z_sim2 = model.feature_extractor(test_input2[..., :5])
                z_real = model.feature_extractor(test_real[..., :5])
            if torch.any(torch.isnan(z_sim1)):
                print(z_sim1)
                raise
            if torch.any(torch.isnan(z_sim2)):
                print(z_sim2)
                raise
            if torch.any(torch.isnan(z_real)):
                print(z_real)
                raise
            latent1 = torch.cat([z_sim1, test_input1[..., 5:]], dim=2)
            latent2 = torch.cat([z_sim2, test_input2[..., 5:]], dim=2)
            pred_shape1 = model.kinematics(latent1)
            pred_shape2 = model.kinematics(latent2)

            kine_loss1 = loss_shape(pred_shape1, test_kine1).mean().item()  # input, target

            kine_loss2 = loss_shape(pred_shape2, test_kine2).mean().item()  # input, target
            pred_domain_s = model.domain_classifier(z_sim1, alpha)  # batch 2
            pred_domain_t = model.domain_classifier(z_real, alpha)  # batch 2

            domain_loss = (loss_classifier(pred_domain_s.to(torch.float32),
                                           torch.tensor([1]).to(torch.float32).to(device)) + \
                           loss_classifier(pred_domain_t.to(torch.float32),
                                           torch.tensor([0]).to(torch.float32).to(device)))
            eval_loss = domain_loss/2 + kine_loss1  +kine_loss2

            writer.add_scalar('Loss/test', eval_loss,epoch)
            writer.add_scalar('Loss/train_domain_diff', train_domain_diff, epoch)
            writer.add_scalar('Loss/train_domain_loss', train_domain_loss, epoch)
            writer.add_scalar('Loss/train_kine1_loss', train_kine1_loss, epoch)
            writer.add_scalar('Loss/train_kine2_loss', train_kine2_loss, epoch)
            writer.add_scalar('Loss/test_domain_loss', domain_loss/2, epoch)
            writer.add_scalar('Loss/test_kine1_loss', kine_loss1, epoch)
            writer.add_scalar('Loss/test_kine2_loss', kine_loss2, epoch)

        if epoch % 50 == 0:
            te = datetime.now().replace(microsecond=0)
            print("Epoch: {:04d}  Domain_diff: {:.5f}| Auto train: {:.8f}|  val: {:.8f}|  Kine1 train: {:.8f}| val: {:.8f}|  Kine2 train: {:.8f}| val: {:.8f}|  time: {}|   Total time: {}|".format(
                epoch,train_domain_diff, train_domain_loss,domain_loss/2,train_kine1_loss,kine_loss1, train_kine2_loss,  kine_loss2,
                (te - ts), te-start_time))
            ts = datetime.now().replace(microsecond=0)
        if kine_loss1 + kine_loss2 < min_kine_loss:
            min_kine_loss = kine_loss1 + kine_loss2
            training_end_num = 0
            torch.save(model.state_dict(),model_path)
        else:
            training_end_num += 1
            if training_end_num > 100:
                break
    return model

####################### MAIN #################################
def main():
    ###################### Set model ###############################
    for run_id in range(5):
        fix_randomness(run_id,train)
        if backbone == 'LSTM' or 'VRNN' or 'CNN':
            model = Build_DANN(backbone)
        model.to(device)
        model_name = backbone +str(run_id)+'_run'+ time.strftime("%Y%m%d-%H%M%S")
        model_path = '../../result/0125/' + model_name + '.pt'
        writer = SummaryWriter('runs/'+model_name)

        print(model_path)
        model = run(model, backbone, model_path, writer)


if __name__ == "__main__":
    main()
