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
####### for server com###########
import sys

from AE_model import POC, Oneshot,fix_randomness
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
train = False
load = False
find_THRESHOLD = False
################## Save path ##################
load_pt_file_name ='.pt' ### NEED TO CHANGE

################## Load data set##################
num_data = 5
from preprocess import AE_preprocess
preprocess = AE_preprocess(dir, 5)
train_dataset1,test_dataset1, train_dataset2, test_dataset2 = preprocess.data_preprocess() ######### LOAD DATASET
length = 15000


def Phase1(model, simAE, x_real, x_sim, optimizer):
    loss_func = nn.MSELoss().to(device) #input, target
    out_sim = model.real2sim(x_real[..., :5]).unsqueeze(0)
    z_sim = simAE.encoder(out_sim)
    latent = torch.cat([z_sim, x_real[..., 5:]], dim=2)
    out_real = model.decoder_real(latent)

    mapping_loss = loss_func(out_sim, x_sim[..., :5])  # input, target
    recon_loss = loss_func(out_real, x_real[..., :5])  # input, target

    loss = mapping_loss + recon_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, [mapping_loss.mean().item(), recon_loss.mean().item()]

def Phase2(model,simAE, num_data,sim_data, optimizer):
    loss_cycle_sum = 0
    for i in range(num_data):
        loss_func = nn.MSELoss().to(device)  # input, target
        z_sim = simAE.encoder(sim_data[i][..., :5])
        latent = torch.cat([z_sim, sim_data[i][..., 5:]], dim=2)

        out_real = model.decoder_real(latent)
        out_sim = model.real2sim(out_real).unsqueeze(0)
        # Cycle loss
        loss_cycle = loss_func(out_sim, sim_data[i][..., :5])

        optimizer.zero_grad()
        loss_cycle.backward()

        optimizer.step()
        loss_cycle_sum += loss_cycle.item()

    return model, [loss_cycle_sum/num_data]

def run(model, simAE, model_path, writer):
    optimizer = torch.optim.Adam([{'params':model.real2sim.parameters(), 'lr':0.0004},\
                                   {'params':model.decoder_real.parameters(), 'lr':0.0004}], weight_decay =1e-6)
    [x_real, x_sim,train_kine1, _] = train_dataset1
    [train_input,_,train_kine2] = train_dataset2
    [test_real, test_input1, test_kine1, _] = test_dataset1
    [test_input2, _, test_kine2] = test_dataset2

    ## 학습하기
    loss_func = nn.MSELoss().to(device) #input, target

    ts = datetime.now().replace(microsecond=0)
    start_time = ts

    training_end_num = 0
    min_eval_loss = 1000
    for epoch in range(90000):
        model.train()
        for i in range(5):
            model, [train_mapping_loss, train_recon_loss] =  Phase1(model,simAE,x_real[1], x_sim[1], optimizer)
            model, [train_loss_cycle] = Phase2(model,simAE, len(x_sim), x_sim, optimizer)
        model.eval()
        with torch.no_grad():
            # Phase 1
            out_sim = model.real2sim(test_real[..., :5]).unsqueeze(0)
            z_sim = simAE.encoder(out_sim)
            latent = torch.cat([z_sim, test_real[..., 5:]], dim=2)
            out_real = model.decoder_real(latent)

            mapping_loss = loss_func(out_sim, test_input1[..., :5]).item()  # input, target
            recon_loss = loss_func(out_real, test_real[..., :5]).item()  # input, target
            eval_loss = mapping_loss + recon_loss

            # Phase 2
            z_sim = simAE.encoder(test_input1[..., :5])
            latent = torch.cat([z_sim, test_input1[..., 5:]], dim=2)
            out_real = model.decoder_real(latent)
            out_sim = model.real2sim(out_real).unsqueeze(0)

            loss_cycle = loss_func(out_sim, test_input1[..., :5]).item()

            writer.add_scalar('Loss/test', eval_loss,epoch)
            writer.add_scalar('Loss/recon_loss', train_recon_loss, epoch)
            writer.add_scalar('Loss/mapping_loss', train_mapping_loss, epoch)
            writer.add_scalar('Loss/loss_cycle', train_loss_cycle, epoch)
            writer.add_scalar('Loss/test_recon_loss', recon_loss, epoch)
            writer.add_scalar('Loss/test_mapping_loss', mapping_loss, epoch)
            writer.add_scalar('Loss/test_loss_cycle', loss_cycle, epoch)
        if epoch % 50 == 0:
            te = datetime.now().replace(microsecond=0)
            print("Epoch: {:04d} test: {:.8f}| Recon train: {:.8f}  val: {:.8f}| Map train: {:.8f}  val: {:.8f}|  Cycle train: {:.8f} val: {:.8f}| time: {}|   Total time: {}|".format(
                epoch,eval_loss,train_recon_loss,recon_loss,train_mapping_loss,mapping_loss, train_loss_cycle,  loss_cycle,
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
def load_model(model, model_path):
    from collections import OrderedDict
    loaded_state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for n, v in loaded_state_dict.items():
        name = n.replace("module.", "")  # .module이 중간에 포함된 형태라면 (".module","")로 치환
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model
            # threshold_score

####################### MAIN #################################
def main():
    super_input, auto_input = 10, 5
    kine_hidden, autoencoder_hidden = 256, 256
    ###################### Set model ###############################

    for run_id in range(5):
        fix_randomness(run_id,train)
        simAE = POC(10, 256, device).to(device)
        simAE = load_model(simAE, os.path.dirname(os.path.abspath(__file__)) + simAE_arr[run_id]+'.pt')

        model = Oneshot(device)
        model.to(device)
        model_name = 'Oneshot_' + str(run_id) + '_run' + time.strftime("%Y%m%d-%H%M%S")
        model_path = 'result/0125/' + model_name + '.pt'
        writer = SummaryWriter('runs/'+model_name)
        print(model_path)
        model = run(model,simAE, model_path, writer)

  
if __name__ == "__main__":
    main()
