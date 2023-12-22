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
from AE_model import DSN, DiffLoss
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import itertools
import random
import torch.backends.cudnn as cudnn
from utils import fix_randomness
deterministic = False # 모델 구현 및 코드 배포시에만 True, 학습에서는 False
cudnn.benchmark = True
import gc
gc.collect()
torch.cuda.empty_cache()
if deterministic:
	cudnn.deterministic = True
	cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train = True
load = False
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
active_domain_loss_step = 10000

################## Load data set##################
num_data = 5
from preprocess import AE_preprocess
preprocess = AE_preprocess()
train_dataset1, test_dataset1, train_dataset2, test_dataset2 = preprocess.data_preprocess() ### LOAD DATASET
# train_dataset3, test_dataset3 = preprocess.kinematics_gait(kinematics3_dir,48,2,device)



################# Loss Calculation #################
loss_classifier = nn.BCEWithLogitsLoss().cuda()  # input, target # for DOMAIN, TASK CLASSIFICATION
loss_MSE = nn.MSELoss().cuda() # input, target # for SHAPE
loss_difference = DiffLoss().cuda()
################## Similarity with MSE
loss_similarity = nn.MSELoss().cuda()
##################
def toCPU(list):
    for i in range(len(list)):
        list[i] = list[i].cpu()
    return list
def train(model, sim=None,real=None, kinematics = None,  mode = None):
    result = []
    if mode == 'task':
        sim, kinematics = sim.cuda(), kinematics.cuda()
        shared_feat1 = model.shared_encoder(sim)
        private_feat1 = model.private_source_encoder(sim)
        latent1 = torch.cat([shared_feat1, sim], dim=2)
        pred_shape = model.kinematics(latent1)
        shape_loss = loss_MSE(pred_shape, kinematics).mean()
        diff_loss = loss_difference(private_feat1, shared_feat1).mean()
        result.append(shape_loss)
        result.append(diff_loss)
        sim, kinematics = sim.cpu(), kinematics.cpu()
        return model , result

    elif mode == 'domain':
        sim, real = sim.cuda(), real.cuda()
        ##### REAL #####
        shared_feat2 = model.shared_encoder(real[..., :5])
        private_feat2 = model.private_target_encoder(real)
        diff_loss2 = loss_difference(private_feat2, shared_feat2).mean()
        union_feat2 = private_feat2 + shared_feat2
        latent2 = torch.cat([union_feat2, real], dim=2)
        decoded_feat2 = model.shared_decoder(latent2)
        recon_loss2 = loss_MSE(decoded_feat2, real).mean()
        ##### SOURCE #####
        shared_feat1 = model.shared_encoder(sim)
        private_feat1 = model.private_source_encoder(sim)
        diff_loss1 = loss_difference(private_feat1, shared_feat1).mean()
        union_feat1 = private_feat1 + shared_feat1
        latent1 = torch.cat([union_feat1, sim], dim=2)
        decoded_feat1 = model.shared_decoder(latent1)
        recon_loss1 = loss_MSE(decoded_feat1, sim).mean()
        ##### LOSS #####
        diff_loss = (diff_loss1 + diff_loss2)/2
        simil_loss = loss_similarity(shared_feat1,shared_feat2).mean()
        recon_loss = (recon_loss1+recon_loss2)/2
        result.append(recon_loss)
        result.append(simil_loss)
        result.append(diff_loss)
        sim, real = sim.cpu(), real.cpu()
        return model , result, [shared_feat1.cpu(), shared_feat2.cpu()]

def run(model, model_path, writer):
    train_dataset1_cpu = toCPU(train_dataset1)
    train_dataset2_cpu = toCPU(train_dataset2)
    test_dataset1_cpu = toCPU(test_dataset1)
    test_dataset2_cpu = toCPU(test_dataset2)

    [x_real, x_sim,train_kine1, _] = train_dataset1_cpu
    [train_input2,_,train_kine2] = train_dataset2_cpu
    [test_real, test_input1, test_kine1, _] = test_dataset1_cpu
    [test_input2, _, test_kine2] = test_dataset2_cpu
    ts = datetime.now().replace(microsecond=0)
    start_time = ts

    params1 = [model.private_target_encoder.parameters(), model.private_source_encoder.parameters(), model.shared_encoder.parameters(), model.shared_decoder.parameters()]
    params2 = [model.private_source_encoder.parameters(),model.shared_encoder.parameters()]
    params3 = [model.kinematics.parameters()]

    optimizer1 = torch.optim.Adam([{'params': itertools.chain(*params1)}], lr= 0.0004, weight_decay = 1e-6) #weight_decay = 1e-6
    optimizer2 = torch.optim.Adam([{'params': itertools.chain(*params2), 'lr':0.0004},{'params': itertools.chain(*params3), 'lr':0.001}], weight_decay = 1e-6)
    training_end_num = 0
    min_kine_loss = 1000
    for epoch in range(90000):
        model.train()
        train_simil_loss, train_diff_loss, train_recon_loss,train_kine1_loss,train_kine2_loss,train_domain_diff = 0, 0, 0, 0,0,0

        for j in range(5):
            for i in range(4):
                loss = 0
                model, [recon_loss, simil_loss, diff_loss], shared_feat = train(model,sim=x_sim[i],real=x_real[i], mode='domain') # recon, simil, diff
                loss += gamma_weight * simil_loss
                loss += beta_weight * diff_loss
                loss += alpha_weight * recon_loss
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                train_domain_diff += torch.mean((shared_feat[0] - shared_feat[1])**2).cpu()
                train_simil_loss += gamma_weight * simil_loss.cpu().item()
                train_diff_loss += beta_weight * diff_loss.cpu().item()
                train_recon_loss += alpha_weight * recon_loss.cpu().item()
        for i in range(len(x_sim)):
            model, [shape_loss,diff_loss] = train(model, sim=x_sim[i],kinematics=train_kine1[i],mode='task')  # shape, diff
            loss = ( shape_loss) + beta_weight * diff_loss
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()

            train_kine1_loss += shape_loss.cpu().item()
            train_diff_loss += beta_weight * diff_loss.cpu().item()

        for i in range(len(train_input2)):
            model.zero_grad()
            model, [shape_loss,diff_loss] = train(model, sim=train_input2[i], kinematics=train_kine2[i],mode='task')
            loss = (shape_loss) + beta_weight * diff_loss
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()

            train_kine2_loss += shape_loss.cpu().item()
            train_diff_loss += beta_weight * diff_loss.cpu().item()
        train_domain_diff = train_domain_diff/20
        [train_simil_loss,train_diff_loss,train_recon_loss]=[train_simil_loss/20,train_diff_loss/36,train_recon_loss/20]
        [train_kine1_loss]=[train_kine1_loss/len(x_sim)]
        [train_kine2_loss]=[train_kine2_loss/len(train_input2)]
        train_domain_loss = train_diff_loss+train_recon_loss
        model.eval()
        with torch.no_grad():
            model, [recon_loss,simil_loss, diff_loss1], _ = train(model, sim =test_input1,real=test_real,mode='domain')
            model, [kine_loss1,diff_loss2] = train(model,sim=test_input1, kinematics=test_kine1,mode='task')

            model, [kine_loss2,diff_loss3] = train(model,sim=test_input2, kinematics=test_kine2, mode='task')

            recon_loss = alpha_weight * recon_loss.item()
            diff_loss= beta_weight * (diff_loss1+diff_loss2+diff_loss3).item()
            simil_loss = gamma_weight * simil_loss.item()
            domain_loss = diff_loss + recon_loss

            eval_loss = recon_loss + diff_loss + simil_loss +kine_loss1 +kine_loss2

            writer.add_scalar('Loss/test', eval_loss,epoch)
            writer.add_scalar('Loss/train_domain_diff', train_domain_diff, epoch)
            writer.add_scalar('Loss/train_simil_loss', train_simil_loss, epoch)
            writer.add_scalar('Loss/train_domain_loss', train_domain_loss, epoch)
            writer.add_scalar('Loss/train_kine1_loss', train_kine1_loss, epoch)
            writer.add_scalar('Loss/train_kine2_loss', train_kine2_loss, epoch)
            writer.add_scalar('Loss/test_domain_loss', domain_loss, epoch)
            writer.add_scalar('Loss/test_simil_loss', simil_loss, epoch)
            writer.add_scalar('Loss/test_kine1_loss', kine_loss1, epoch)
            writer.add_scalar('Loss/test_kine2_loss', kine_loss2, epoch)

        if epoch % 50 == 0:
            te = datetime.now().replace(microsecond=0)
            print("Epoch: {:04d}  Domain_diff: {:.5f}| Simil_diff: {:.5f}| Auto train: {:.8f}|  val: {:.8f}|  Kine1 train: {:.8f}| val: {:.8f}|  Kine2 train: {:.8f}| val: {:.8f}|  time: {}|  Total time: {}|".format(
                epoch,train_domain_diff,simil_loss, train_domain_loss,domain_loss,train_kine1_loss,kine_loss1, train_kine2_loss,  kine_loss2,
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
        model = DSN()
        model.to(device)
        model_name = 'DSN' +str(run_id)+'_run'+ time.strftime("%Y%m%d-%H%M%S")
        model_path = '../../result/0125/' + model_name + '.pt'

        writer = SummaryWriter('runs/'+model_name)
        print('writer created')
        print(model_path)
        model = run(model, model_path, writer)

if __name__ == "__main__":
    main()
