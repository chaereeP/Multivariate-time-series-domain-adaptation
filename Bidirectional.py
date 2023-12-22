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
import sys
from AE_model import Bidirectional, fix_randomness
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import itertools
import random
import torch.backends.cudnn as cudnn

torch.set_flush_denormal(True)
deterministic = False # 모델 구현 및 코드 배포시에만 True, 학습에서는 False
cudnn.benchmark = True
if deterministic:
    cudnn.deterministic = True
    cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
AE_mode = 'nosingle'
pressure_mode = 'pressure'

num_data = 5

from preprocess import AE_preprocess
preprocess = AE_preprocess()
train_dataset1,test_dataset1, train_dataset2, test_dataset2 = preprocess.data_preprocess() # LOAD DATASET

length = 15000
def Phase1_unob(model, num_data,x_sim, kinematics,optimizer2):
    train_kine_loss, train_sim_loss = 0,0
    loss_func = nn.MSELoss().to(device) #input, target
    for i in range(num_data):
        z_sim = model.autoencoder.encoder_sim(x_sim[i][..., :5])
        latent = torch.cat([z_sim, x_sim[i][..., 5:]], dim=2)
        x_outsim = model.autoencoder.reconstruct_decoder_sim(latent)
        output_kinematics = model.kinematics(z_sim)
        kine_loss = loss_func(output_kinematics, kinematics[i])  # input, target
        recon_loss = loss_func(x_outsim, x_sim[i][..., :5])  # input, target
        loss = kine_loss + recon_loss
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        train_kine_loss += kine_loss.mean().item()
        train_sim_loss += recon_loss.mean().item()
    return model, [train_kine_loss/num_data, train_sim_loss/num_data]

def Phase1_ob(model, num_data,x_sim, kinematics,optimizer2):
    train_kine_loss = 0
    loss_func = nn.MSELoss().to(device) #input, target
    for i in range(num_data):
        z_sim = model.autoencoder.encoder_sim(x_sim[i][..., :5])
        output_kinematics = model.kinematics(z_sim)
        if kinematics[i].shape != output_kinematics.shape:
            raise
        kine_loss = loss_func(output_kinematics, kinematics[i])  # input, target
        loss = kine_loss
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        train_kine_loss += kine_loss.mean().item()
    return model, [train_kine_loss/num_data]

def backward_G(model,real_data, sim_data, optimizer_s, optimizer_r):
    loss_func = torch.nn.L1Loss().to(device)
    enc_a = model.autoencoder.encoder_real(real_data[..., :5])
    latent_r = torch.cat([enc_a, real_data[..., 5:]], dim=2)
    fake_AA = model.autoencoder.reconstruct_decoder_real(latent_r)
    fake_AB = model.autoencoder.reconstruct_decoder_sim(latent_r)
    enc_ab = model.autoencoder.encoder_sim(fake_AB)
    latent_rs = torch.cat([enc_ab, sim_data[..., 5:]], dim=2)
    fake_ABA = model.autoencoder.reconstruct_decoder_real(latent_rs)
    loss_cycle_A = loss_func(fake_ABA, real_data[...,  :5])
    loss_idt_A = loss_func(fake_AA, real_data[..., :5])
    enc_b = model.autoencoder.encoder_sim(sim_data[..., :5])
    latent_s = torch.cat([enc_b, sim_data[..., 5:]], dim=2)
    fake_BB = model.autoencoder.reconstruct_decoder_sim(latent_s)
    fake_BA = model.autoencoder.reconstruct_decoder_real(latent_s)
    enc_ba = model.autoencoder.encoder_real(fake_BA)
    latent_sr = torch.cat([enc_ba, real_data[..., 5:]], dim=2)
    fake_BAB = model.autoencoder.reconstruct_decoder_sim(latent_sr)
    loss_idt_B = loss_func(fake_BB, sim_data[...,  :5])
    loss_cycle_B = loss_func(fake_BAB, sim_data[...,  :5])
    loss_feat_BA = 0.001* loss_func(enc_ba, enc_b.detach()) +0.001* loss_func(enc_a, enc_b.detach())

    loss_G_A = loss_cycle_A + loss_idt_A + loss_feat_BA
    loss_G_B = loss_cycle_B + loss_idt_B
    return loss_G_A, loss_G_B

def backward_G_sim(model,sim_data, optimizer_s, optimizer_r):
    loss_func = torch.nn.L1Loss().to(device)
    enc_b = model.autoencoder.encoder_sim(sim_data[..., :5])
    latent_s = torch.cat([enc_b, sim_data[..., 5:]], dim=2)
    fake_BB = model.autoencoder.reconstruct_decoder_sim(latent_s)
    fake_BA = model.autoencoder.reconstruct_decoder_real(latent_s)
    enc_ba = model.autoencoder.encoder_real(fake_BA)
    latent_sr = torch.cat([enc_ba, sim_data[..., 5:]], dim=2)
    fake_BAB = model.autoencoder.reconstruct_decoder_sim(latent_sr)
    loss_idt_B = loss_func(fake_BB, sim_data[...,  :5])
    loss_cycle_B = loss_func(fake_BAB, sim_data[...,  :5])
    loss_feat_BA = 0.001* loss_func(enc_ba, enc_b.detach())
    loss_G_B = loss_cycle_B + loss_idt_B
    return loss_G_B

def Phase2_unob(model, num_data,real_data, sim_data, optimizer_s, optimizer_r):
    loss_G_A_sum, loss_G_B_sum = 0,0
    loss_G_A, loss_G_B = backward_G(model, real_data[0], sim_data[0], optimizer_s, optimizer_r)
    optimizer_r.zero_grad()
    loss_G_A.backward(retain_graph=True)

    # B loss updates
    optimizer_s.zero_grad()
    loss_G_B.backward()

    optimizer_r.step()
    optimizer_s.step()

    loss_G_A_sum += loss_G_A.item()
    loss_G_B_sum += loss_G_B.item()

    for i in range(num_data):
        loss_G_B = backward_G_sim(model, sim_data[i], optimizer_s, optimizer_r)

        # B loss updates
        optimizer_s.zero_grad()
        loss_G_B.backward()
        optimizer_s.step()
        loss_G_B_sum += loss_G_B.item()

    return model, [loss_G_A_sum, loss_G_B_sum/(num_data+1)]

def Phase2_kine(model, num_data, x_unob, x_ob, kinematics1,kinematics2, optimizer2):
    train_kine1_loss, train_kine2_loss = 0,0
    loss_func = nn.MSELoss().to(device) #input, target
    for i in range(num_data):
        z_sim = model.autoencoder.encoder_sim(x_unob[i][..., :5])
        output_kinematics = model.kinematics(z_sim)
        if kinematics1[i].shape != output_kinematics.shape:
            raise
        kine_loss = loss_func(output_kinematics, kinematics1[i])  # input, target
        loss = kine_loss
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        train_kine1_loss += kine_loss.mean().item()
    for i in range(num_data):
        z_sim = model.autoencoder.encoder_sim(x_ob[i][..., :5])
        output_kinematics = model.kinematics(z_sim)
        if kinematics2[i].shape != output_kinematics.shape:
            raise
        kine_loss = loss_func(output_kinematics, kinematics2[i])  # input, target
        loss = kine_loss
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        train_kine2_loss += kine_loss.mean().item()
    return model, [train_kine1_loss/num_data, train_kine2_loss/num_data]
def initial_weighting(model,model_path):
    # Load the weights
    from collections import OrderedDict
    weights = torch.load(model_path)
    # Make a copy of the weights
    weights_copy = {k: v.clone() for k, v in weights.items()}
    real = ['reconstruct_decoder_real','encoder_real']
    sim = ['reconstruct_decoder_sim','encoder_sim']
    lstm = ['.lstm1.','.lstm2.']
    lstm_weight = ['weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0' ]
    reconst_real = ['reconstruct_decoder_real.fc.weight', 'reconstruct_decoder_real.fc.bias']
    reconst_sim = ['reconstruct_decoder_sim.fc.weight', 'reconstruct_decoder_sim.fc.bias']
    # Initialize your model with the copy of the weights
    new_state_dict = OrderedDict()
    for i in range(2):
        for k in range(2):
            for j in range(4):
                sim_name = 'autoencoder.'+sim[i] + lstm[k] + lstm_weight[j]
                real_name = 'autoencoder.'+real[i] + lstm[k] + lstm_weight[j]
                new_state_dict[real_name] = weights_copy[sim_name]
                new_state_dict[sim_name] = weights_copy[sim_name]

        new_state_dict['autoencoder.'+reconst_real[i]] = weights_copy['autoencoder.'+reconst_sim[i]]
        new_state_dict['autoencoder.'+reconst_sim[i]] = weights_copy['autoencoder.'+reconst_sim[i]]

    for n, v in weights_copy.items():
        if 'autoencoder' not in n:
            name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model
def run(model, model_path, writer):
    lr1 = 0.0004#0.0004
    lr2 = 0.001#0.001

    # optimizer1 = torch.optim.Adam(model.autoencoder.parameters(), lr=lr1, weight_decay = 1e-6)
    optimizer1 = torch.optim.Adam([{'params':model.kinematics.parameters(), 'lr':0.001}, \
            {'params':model.autoencoder.encoder_sim.parameters(), 'lr':0.0004},\
                                   {'params':model.autoencoder.reconstruct_decoder_sim.parameters(), 'lr':0.0004}], weight_decay =1e-6)
    optimizer_s = torch.optim.Adam([{'params':model.autoencoder.encoder_sim.parameters(), 'lr':0.0004},\
                                   {'params':model.autoencoder.reconstruct_decoder_sim.parameters(), 'lr':0.0004}], weight_decay =1e-6)
    optimizer_r = torch.optim.Adam([{'params': model.autoencoder.encoder_real.parameters(), 'lr': 0.0004}, \
                                    {'params': model.autoencoder.reconstruct_decoder_real.parameters(), 'lr': 0.0004}],
                                   weight_decay=1e-6)
    [x_real, x_sim,train_kine1, _] = train_dataset1
    [train_input,_,train_kine2] = train_dataset2
    # [train_input3,train_collision3,train_kine3] = train_dataset3

    [test_real, test_input1, test_kine1, _] = test_dataset1
    [test_input2, _, test_kine2] = test_dataset2

    ## 학습하기
    loss_func = nn.MSELoss().to(device) #input, target

    ts = datetime.now().replace(microsecond=0)
    start_time = ts

    training_end_num = 0
    min_kine_loss = 1000
    min_eval_loss = 1000
    switch = 0
    for epoch in range(90000):
        model.train()
        if not switch :
            model, [train_kine1_loss, train_sim_loss] = Phase1_unob(model, len(x_sim), x_sim, train_kine1, optimizer1)
            model, [train_kine2_loss] = Phase1_ob(model, len(x_sim), x_sim, train_kine2, optimizer1)
            model.eval()
            with torch.no_grad():
                loss = 0
                z_sim = model.autoencoder.encoder_sim(test_input1[..., :5])
                latent = torch.cat([z_sim, test_input1[..., 5:]], dim=2)
                x_outsim = model.autoencoder.reconstruct_decoder_sim(latent)
                output_kinematics1 = model.kinematics(z_sim)

                sim_loss = loss_func(test_input1[:, :, :5], x_outsim).item()
                loss += sim_loss
                kine_loss1 = loss_func(output_kinematics1, test_kine1)

                z_sim2 = model.autoencoder.encoder_sim(test_input2[..., :5])
                output_kinematics2 = model.kinematics(z_sim2)
                kine_loss2 = loss_func(output_kinematics2, test_kine2).item()

                if test_kine2.shape != output_kinematics2.shape or test_kine1.shape != output_kinematics1.shape:
                    raise
                eval_loss = loss
                writer.add_scalar('Loss/test', eval_loss, epoch)
                writer.add_scalar('Loss/train_sim_loss', train_sim_loss, epoch)
                writer.add_scalar('Loss/train_kine1_loss', train_kine1_loss, epoch)
                writer.add_scalar('Loss/train_kine2_loss', train_kine2_loss, epoch)
                writer.add_scalar('Loss/test_sim_loss', sim_loss, epoch)
                writer.add_scalar('Loss/test_kine1_loss', kine_loss1, epoch)
                writer.add_scalar('Loss/test_kine2_loss', kine_loss2, epoch)
            if epoch % 50 == 0:
                te = datetime.now().replace(microsecond=0)
                print(
                    "Epoch: {:04d}  Phase {:01d}| Auto train: {:.8f}|  val: {:.8f}|  Kine1 train: {:.8f}| val: {:.8f}|  Kine2 train: {:.8f}| val: {:.8f}|   time: {}|   Total time: {}|".format(
                        epoch, switch,train_sim_loss, eval_loss, train_kine1_loss, kine_loss1, train_kine2_loss, kine_loss2,
                        (te - ts), te - start_time))
                ts = datetime.now().replace(microsecond=0)
            if (kine_loss1 + kine_loss2) < min_kine_loss:
                if (kine_loss1 + kine_loss2)< min_kine_loss:
                    min_kine_loss = kine_loss1 + kine_loss2
                training_end_num = 0
                torch.save(model.state_dict(),model_path)
                model = initial_weighting(model, model_path)
            else:
                training_end_num += 1
                if training_end_num > 100:
                    model = initial_weighting(model, model_path)
                    print("############ SWITCHED #################")
                    min_eval_loss = 1000
                    training_end_num = 0
                    switch = 1
        else:
            model, [loss_G_A_sum, loss_G_B_sum] =  Phase2_unob(model, len(x_sim),x_real, x_sim, optimizer_s, optimizer_r)

            model, [train_kine1_loss, train_kine2_loss] =  Phase2_kine(model, 4, x_sim, train_input,train_kine1, train_kine2, optimizer1)
            model.eval()
            with torch.no_grad():
                loss= 0
                z_sim1,z_real, x_outsim, x_outreal  = model.autoencoder.anomaly(test_input1, test_real)
                real_loss = loss_func(test_real[:, :, :5], x_outreal).item()
                loss += real_loss
                sim_loss = loss_func(test_input1[:,:,:5], x_outsim).item()
                loss += sim_loss
                feature_loss = loss_func(z_sim1, z_real).item()
                loss +=  feature_loss
                output_kinematics1 = model.kinematics(z_sim1)
                kine_loss1 = loss_func(output_kinematics1,test_kine1)
                z_sim2 = model.autoencoder.encoder_sim(test_input2[..., :5])
                output_kinematics2 = model.kinematics(z_sim2)
                kine_loss2 = loss_func(output_kinematics2,test_kine2).item()

                if test_kine2.shape != output_kinematics2.shape or test_kine1.shape != output_kinematics1.shape :
                    raise
                eval_loss = loss
                writer.add_scalar('Loss/test', eval_loss,epoch)
                writer.add_scalar('Loss/loss_G_B_sum', loss_G_B_sum, epoch)
                writer.add_scalar('Loss/loss_G_A_sum', loss_G_A_sum, epoch)
                writer.add_scalar('Loss/train_kine1_loss', train_kine1_loss, epoch)
                writer.add_scalar('Loss/train_kine2_loss', train_kine2_loss, epoch)
                writer.add_scalar('Loss/test_sim_loss', sim_loss, epoch)
                writer.add_scalar('Loss/test_feature_loss', feature_loss, epoch)
                writer.add_scalar('Loss/test_kine1_loss', kine_loss1, epoch)
                writer.add_scalar('Loss/test_kine2_loss', kine_loss2, epoch)
            if epoch % 50 == 0:
                te = datetime.now().replace(microsecond=0)
                print("Epoch: {:04d} G_B train: {:.8f} G_B train: {:.8f}| Auto val: {:.8f}|  Kine1 train: {:.8f}| val: {:.8f}|  Kine2 train: {:.8f}| val: {:.8f}|   time: {}|   Total time: {}|".format(
                    epoch,loss_G_B_sum,loss_G_A_sum,eval_loss,train_kine1_loss,kine_loss1, train_kine2_loss,  kine_loss2,
                    (te - ts), te-start_time))
                ts = datetime.now().replace(microsecond=0)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                training_end_num = 0
                torch.save(model.state_dict(),model_path)
            else:
                training_end_num += 1
                if training_end_num > 100:
                    break
    return model
  
def load_model(model, model_path): ### if you want to test, you can load model with this function.
    from collections import OrderedDict
    loaded_state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

####################### MAIN #################################
def main():
    super_input, auto_input = 5, 5
    kine_hidden, autoencoder_hidden = 256, 256
    ###################### train ###############################
    for run_id in range(5):
        fix_randomness(run_id,train)
        model = Bidirectional(pressure_mode, AE_mode, super_input, auto_input, kine_hidden, autoencoder_hidden, device)
        model.to(device)
        if pressure_mode == 'nopressure':
            model_name = 'Bidirectional_'  + pressure_mode+str(run_id) + '_run' + time.strftime("%Y%m%d-%H%M%S")
        else:
            model_name = 'Bidirectional_'  + str(run_id) + '_run' + time.strftime("%Y%m%d-%H%M%S")
        model_path = 'result/0611/' + model_name + '.pt'
        writer = SummaryWriter('runs/'+model_name)
        print(model_path)
        model = run(model, model_path, writer)


if __name__ == "__main__":
    main()
