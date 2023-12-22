from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
global device
from torch.autograd import Function
import random
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.hidden_size = 5
        self.lstm1 = nn.LSTM(input_size, hidden_dim, 1,batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, self.hidden_size, 1,batch_first=True)
        # self.m = nn.BatchNorm1d(self.hidden_size)#, affine=False

    def forward(self, x):
        device = 'cuda'
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))
        h_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(device))
        c_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(device))
        outputs, (_, _) = self.lstm1(x,(h_0, c_0))
        outputs, (_, _) = self.lstm2(outputs,(h_1, c_1))
        # outputs = outputs.reshape(1,self.hidden_size,-1)
        # outputs=self.m(outputs)
        outputs = outputs.reshape(1,-1,self.hidden_size)
        # self.hidden =h_out
        return outputs

class Decoder(nn.Module):
    def __init__(self,  input_size, output_size, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.hidden_size = 5
        self.output_size = output_size
        self.input_size = input_size

        self.lstm1 = nn.LSTM(self.input_size, hidden_dim, 1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True,bidirectional=False)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        device = 'cuda'
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))
        h_1 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))
        c_1 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).to(device))

        x, (_, _)  = self.lstm1(x,(h_0, c_0))
        x, (_, _)  = self.lstm2(x,(h_1, c_1))
        output = self.fc(self.relu(x))
        # output = self.fc(x)

        return output


class cycleGAN(nn.Module):
    def __init__(self, device_):
        super().__init__()
        global device
        device = device_
        self.G = Generator().to(device)
        self.F = Generator().to(device)

        self.Dx = Discriminator().to(device)
        self.Dy = Discriminator().to(device)

class Seq2Seq(nn.Module):
    def __init__(self,pressure_mode, input_size,hidden_size, device_):
        super().__init__()
        global device
        device = device_
        output_size = 5
        self.input_size = input_size

        self.encoder_sim = Encoder(
            self.input_size,
            hidden_size
        ).to(device)
        self.pressure_mode = pressure_mode
        if pressure_mode =='pressure':
            decoder_size = 10
        else: decoder_size=5
        self.reconstruct_decoder_sim = Decoder(decoder_size,
            output_size,
            hidden_size
        ).to(device)
        self.reconstruct_decoder_real = Decoder(decoder_size,
                                                output_size,
                                                hidden_size
                                                ).to(device)
        self.encoder_real = Encoder(
            self.input_size,
            hidden_size
        ).to(device)

    ## Loss 출력
    def forward(self, x_sim, x_real):
        # src: tensor of shape (batch_size, seq_length, hidden_size)
        # trg: tensor of shape (batch_size, seq_length, hidden_size)
        ## Encoder 넣기
        z_sim = self.encoder_sim(x_sim)
        z_real = self.encoder_real(x_real)
        ## Reconstruction Loss 계산
        x_outsim = self.reconstruct_decoder_sim(z_sim)
        x_outreal = self.reconstruct_decoder_real(z_real)
        return z_sim,z_real, x_outsim , x_outreal

    def decoder_sim(self, data):
        with torch.no_grad():
            data = data.reshape(1, -1, 5) # batch seq feature
            z_data = self.encoder_sim(data)
            out_data = self.reconstruct_decoder_sim(z_data)
        return out_data

    def anomaly(self, x_sim, x_real):
        z_sim = self.encoder_sim(x_sim[..., :5])
        z_real = self.encoder_real(x_real[..., :5])
        ## Reconstruction Loss 계산
        if self.pressure_mode == 'pressure':
            latent1 = torch.cat([z_sim, x_sim[..., 5:]], dim=2)
            latent2 = torch.cat([z_real, x_real[..., 5:]], dim=2)
        else:
            latent1 =z_sim
            latent2 =z_real

        x_outsim = self.reconstruct_decoder_sim(latent1)
        x_outreal = self.reconstruct_decoder_real(latent2)
        return z_sim,z_real, x_outsim , x_outreal

class SingleAE(nn.Module):
    def __init__(self,pressure_mode, input_size,hidden_size, device_):
        super().__init__()
        global device
        device = device_
        output_size = 5
        input_size = 5
        self.encoder = Encoder(
            input_size,
            hidden_size
        ).to(device)
        decoder_input = 5
        self.pressure_mode= pressure_mode
        if pressure_mode == 'pressure':
            decoder_input = 10
        self.decoder = Decoder(decoder_input,
            output_size,
            hidden_size
        ).to(device)
    ## Loss 출력
    def anomaly(self, x_sim, x_real):
        z_sim = self.encoder(x_sim[..., :5])
        z_real = self.encoder(x_real[..., :5])
        if self.pressure_mode == 'pressure':
            latent1 = torch.cat([z_sim, x_sim[..., 5:]], dim=2)
            latent2 = torch.cat([z_real, x_real[..., 5:]], dim=2)
        else:
            latent1= z_sim
            latent2= z_real

        x_outsim = self.decoder(latent1)
        x_outreal = self.decoder(latent2)
        return z_sim,z_real, x_outsim , x_outreal

class LSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size, device):
        super(LSTM, self).__init__()
        self.device = device
        self.num_layers = num_layers
        input_size = 5
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.i2o = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.functional.tanh(hidden_size)
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
        # Propagate input through LSTM

        output, (h_out, _) = self.lstm(input, (h_0, c_0))

        output = output.view(-1, self.hidden_size)
        out = self.i2o(output)
        # out= torch.nn.functional.tanh(out)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.linear2(out)
        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



class LSTM2(nn.Module): # kinnematics
    def __init__(self, input_size, num_layers, hidden_size, output_size,device):
        super(LSTM2, self).__init__()
        self.device=device
        self.num_layers=num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1,batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1,batch_first=True)
        # self.i2o = nn.Linear(hidden_size , hidden_size)
        # self.relu = nn.functional.tanh(hidden_size)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(hidden_size,output_size)
        # self.relu2 = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_size,output_size)


    def forward(self, input):
        h_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(self.device))
        # h_1 = Variable(torch.zeros(self.num_layers, self.hidden_size, self.hidden_size).to(self.device))
        # c_1 = Variable(torch.zeros(self.num_layers, self.hidden_size, self.hidden_size).to(self.device))
        # Propagate input through LSTM
        # batch seq feature

        output, (h_out, _) = self.lstm1(input, (h_0, c_0))
        output = output.view(-1, self.hidden_size)
        # out= torch.nn.functional.tanh(output)
        out =self.relu(output)
        out=self.linear1(out)
        # out=self.relu2(out)
        return out

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM3(nn.Module): # sim_to_real_mapping
    def __init__(self, input_size, num_layers, hidden_size, output_size,device):
        super(LSTM3, self).__init__()
        self.device=device
        self.num_layers=num_layers

        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.output_size=output_size
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        global device
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        h_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        c_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))

        outputs, (_, _) = self.lstm1(x, (h_0, c_0))
        outputs, (_, _) = self.lstm2(outputs, (h_1, c_1))
        outputs = outputs.view(-1, self.hidden_size)

        outputs =self.relu(outputs)
        outputs=self.linear1(outputs)
        # if self.normalize == 'layer':
        #     outputs=self.m(outputs)
        return outputs

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def predictor(model, sim, real, device):
    predictions, losses = [], []
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        model = model.eval()
        sim = sim.to(device)
        real = real.to(device)
        import numpy as np
        for i in range(len(sim)):
            z_sim, z_real,x_outsim, x_outreal = model(sim[i].reshape(1,-1,10), real[i].reshape(1,-1,10))
            x_outreal = x_outreal.cpu().numpy()#.reshape(-1,)
            x_outsim = x_outsim.cpu().numpy()

            predictions.append(x_outsim)
            losses.append( np.abs(x_outsim- sim[i][...,:5].cpu().numpy()))

            # losses.append(loss2.item())

    return losses


def fit_norm_distribution_param( model, train_dataset, channel_idx=0, device = 'cuda'):
    with torch.no_grad():
        out = model.decoder_sim(train_dataset)
    losses = train_dataset[ ..., channel_idx] - out[ ..., channel_idx]
    losses = losses.reshape(-1,1)
    mean = losses.mean(dim=0)
    cov = losses.t().mm(losses)/losses.size(0) - mean.view(-1,1).mm(mean.unsqueeze(0))
    # cov: positive-semidefinite and symmetric.

    return mean, cov

def get_precision_recall(score, label, num_samples, beta=1.0, sampling='log', predicted_score=None, device = 'cuda'):
    '''
    :param score: anomaly scores
    :param label: anomaly labels
    :param num_samples: the number of threshold samples
    :param beta:
    :param scale:
    :return:
    '''
    # beta = 0.5
    if predicted_score is not None:
        score = score - torch.FloatTensor(predicted_score).squeeze().to(device)

    maximum = score.max()
    if sampling=='log':
        # Sample thresholds logarithmically
        # The sampled thresholds are logarithmically spaced between: math:`10 ^ {start}` and: math:`10 ^ {end}`.
        th = torch.logspace(0, torch.log10(maximum.clone().detach()), num_samples).to(device)
    else:
        # Sample thresholds equally
        # The sampled thresholds are equally spaced points between: attr:`start` and: attr:`end`
        th = torch.linspace(0, maximum, num_samples).to(device)

    precision = []
    recall = []
    for i in range(len(th)):
        anomaly = (score > th[i]).float()
        idx = anomaly * 2 + label
        tn = (idx == 0.0).sum().item()  # tn
        fn = (idx == 1.0).sum().item()  # fn
        fp = (idx == 2.0).sum().item()  # fp
        tp = (idx == 3.0).sum().item()  # tp

        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)

        if p != 0 and r != 0:
            precision.append(p)
            recall.append(r)

    precision = torch.FloatTensor(precision)
    recall = torch.FloatTensor(recall)
    f1 = (1 + beta ** 2) * (precision * recall).div(beta ** 2 * precision + recall + 1e-7)

    return precision, recall, f1

class Supervision_collision(nn.Module):
    def __init__(self,input_size,hidden_size,device):
        super(Supervision_collision, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        w1 = self.l1(x)
        w2 = self.l2(w1)

        return w1,w2

class Supervision_kinematics(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Supervision_kinematics, self).__init__()
        self.hidden_size = hidden_size
        self.device = 'cuda'
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.dout = nn.Dropout(0.1)
        self.l2 = nn.Linear(hidden_size, 369)
        # self.l3 = nn.Linear(hidden_size, 369)


    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        # h_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        # c_1 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        # batch seq feature
        x, (h_out, _) = self.lstm1(x, (h_0, c_0))
        # x, (h_out, _) = self.lstm2(x, (h_1, c_1))
        x = x.view(-1, self.hidden_size)
        # out= torch.nn.functional.tanh(output)
        x =self.relu(x)
        # x = self.dout(x)
        x=self.relu(self.l1(x))
        x=self.tanh(self.l2(x))
        # x=self.relu(self.l3(x))
        return x
class LSTM_prediction(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTM_prediction, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        # self.lstm2 = nn.LSTM(hidden_size, 5, 1, batch_first=True)
        self.l1 = nn.Linear(hidden_size, 5)

        self.relu = nn.ReLU()
        self.dout = nn.Dropout(0.1)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size).to(self.device))
        x, (h_out, _) = self.lstm1(x, (h_0, c_0))
        x = self.dout(h_out)
        x = self.l1(x)
        x = self.relu(x)
        return x
    def predict(self,x):
        predicted_arr = []
        for i in range(len(x[0])-100):
            predicted = self.forward(x[0,i:100+i,:].reshape(1,-1,10))
            predicted_arr.append(predicted)
        return predicted_arr

class MMD_Loss(nn.Module):
    def __init__(self, kernel='rbf'):
        super(MMD_Loss, self).__init__()

    def gaussian_kernel(self, source, target, sigma=1.0): #gaussian_kernel
        n, m = source.size(0), target.size(0)
        source = source.unsqueeze(1).expand(n, m, -1)
        target = target.unsqueeze(0).expand(n, m, -1)
        kernel_val = torch.exp(-0.5 * (source - target) ** 2 / (sigma ** 2))
        return torch.mean(kernel_val, dim=-1)

    def forward(self, source, target, sigma=1.0):
        source_kernel = gaussian_kernel(source, source, sigma)
        target_kernel = gaussian_kernel(target, target, sigma)
        cross_kernel = gaussian_kernel(source, target, sigma)
        loss = source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()
        return loss

class LSTM_MMD(nn.Module):
    def __init__(self, hidden_dim, decive):
        super(LSTM_MMD, self).__init__()
        output_size = 5
        self.layer = Encoder( 5, hidden_dim ).to(device)
        self.mmd = MMD_Loss()

    def forward(self, source, target):
        source = self.layer(source)
        target = self.layer(target)
        mmd_loss = self.mmd(source, target)
        return mmd_loss

class Supervision(nn.Module):
    def __init__(self,AE_mode, Recon_mode,super_input,auto_input, kine_hidden,collision_hidden, autoencoder_hidden, device):
        super().__init__()
        self.kinematics = Supervision_kinematics(super_input, kine_hidden, device).to(device)
        self.collision = Supervision_collision(super_input, collision_hidden, device).to(device)
        pressure_mode = 'nopressure'
        self.autoencoder = Seq2Seq(pressure_mode, auto_input,autoencoder_hidden, device).to(device)


def roc_curve_plot(y_true , y_score ):
    # https://datascienceschool.net/03%20machine%20learning/09.04%20%EB%B6%84%EB%A5%98%20%EC%84%B1%EB%8A%A5%ED%8F%89%EA%B0%80.html

    # 임곗값에 따른 FPR, TPR 값을 반환 받음.
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    # fprs, tprs, thresholds = metrics.roc_curve(y_true, y_score)
    # fprs, tprs, thresholds = metrics.roc_auc_score(y_true, y_score)
    #### START of AUPRC curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    precision = list(precision)
    recall = list(recall)
    thresholds = list(thresholds)

    precision.reverse()
    recall.reverse()
    thresholds.reverse()
    sklearn_AUPRC  = metrics.average_precision_score(y_true, y_score)
    print(sklearn_AUPRC)

    fig = plt.figure()
    fig.set_size_inches(15, 15)
    plt.plot(recall, precision)
    plt.fill_between(recall, precision, 0, facecolor="red", alpha=0.2)
    plt.xlabel("recall", fontsize=24)
    plt.ylabel("precision", fontsize=24)
    for i in range(len(thresholds)):
        plt.text(recall[i], precision[i], thresholds[i], fontsize=18)
    plt.text(recall[-1] / 2, precision[-1] / 2, 'AUPRC : ' + str(sklearn_AUPRC), fontsize=24)
    #### END of AUPRC curve


    conf_mat  = metrics.confusion_matrix(y_true, y_score<-0.178)
    print(conf_mat)
    labels = ['class 0', 'class 1']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    plt.show()
    p = metrics.precision_score(y_true, y_score>-0.178)
    print(p)
    r = metrics.recall_score(y_true, y_score>-0.178)
    print(r)
    f1 = metrics.f1_score(y_true, y_score>-0.178)
    print(f1)

    # from sklearn.metrics import *
    plt.plot(thresholds)
    plt.show()
    # ROC Curve를 plot 곡선으로 그림.
    plt.plot(fprs, tprs, label='ROC')
    # 가운데 대각선 직선을 그림.
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()
    #confusion_matrix
    #accuracy_score,precision_score,recall_score,fbeta_score,f1_score


class Supervision_anomaly(nn.Module):
    def __init__(self,pressure_mode, AE_mode,super_input,auto_input, kine_hidden, autoencoder_hidden, device):
        super().__init__()
        self.kinematics = Supervision_kinematics( super_input, kine_hidden, device).to(device)
        if AE_mode =='nosingle':
            self.autoencoder = Seq2Seq(pressure_mode, auto_input ,autoencoder_hidden, device).to(device)
        elif AE_mode =='single':
            self.autoencoder = SingleAE(pressure_mode, auto_input,autoencoder_hidden, device).to(device)

class Supervision_anomalkine_noP(nn.Module):
    def __init__(self,pressure_mode, AE_mode,super_input,auto_input, kine_hidden, autoencoder_hidden, device):
        super().__init__()
        self.kinematics = Supervision_kinematics( 5, kine_hidden, device).to(device)
        if AE_mode =='nosingle':
            self.autoencoder = Seq2Seq(pressure_mode, auto_input ,autoencoder_hidden, device).to(device)
        elif AE_mode =='single':
            self.autoencoder = SingleAE(pressure_mode, auto_input,autoencoder_hidden, device).to(device)

class Bidirectional(nn.Module):
    def __init__(self,pressure_mode, AE_mode,super_input,auto_input, kine_hidden, autoencoder_hidden, device):
        super().__init__()
        self.kinematics = Supervision_kinematics( super_input, kine_hidden, device).to(device)
        self.autoencoder = Seq2Seq(pressure_mode, auto_input, autoencoder_hidden, device).to(device)

class Oneshot(nn.Module):
    def __init__(self,device):
        super().__init__()
        # self.kinematics = Supervision_kinematics( super_input, kine_hidden, device).to(device)

        # self.encoder = Encoder(input_size,hidden_size).to(device)
        # self.decoder_sim = Decoder(decoder_input,output_size,hidden_size).to(device)
        self.decoder_real = Decoder(10,5,256).to(device)
        self.real2sim = LSTM3(5, 1, 256, 5, device).to(device)

class POC(nn.Module):
    def __init__(self, super_input, kine_hidden, device):
        super().__init__()
        self.kinematics = Supervision_kinematics(super_input, kine_hidden, device).to(device)
        self.encoder = Encoder(
            5,
            256
        ).to(device)
        self.decoder = Decoder(10,
            5,
            256
        ).to(device)

class VRNN(nn.Module):
    def __init__(self):
        super(VRNN, self).__init__()

        self.x_dim = 5
        self.h_dim = 256
        self.z_dim = 5
        self.n_layers = 1

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.h_dim, 1, bias=False)


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_z_t = torch.zeros(3000, self.z_dim, device='cuda')
        kld_loss = 0
        nll_loss = 0
        x = x.reshape(1,-1, 5)
        h = torch.zeros(1, x.size(0), self.h_dim, device='cuda') # 1, 5, 256
        for t in range(x.size(1)): # step_size * feature_size

            phi_x_t = self.phi_x(x[:, t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1)) # -1
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            #nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            nll_loss += self._nll_bernoulli(dec_mean_t, x[:,t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            all_z_t[t]=z_t
        return kld_loss/x.size(1), nll_loss/x.size(1), \
            (all_enc_mean, all_enc_std), \
            (all_dec_mean, all_dec_std), \
               all_z_t.unsqueeze(0)
        # torch.mean(kld_loss), torch.mean(nll_loss)

    def sample(self, seq_len):
        device= 'cuda'
        sample = torch.zeros(seq_len, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        device = 'cuda'
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        EPS = 1e-9
        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        EPS = 1e-9
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        EPS = 1e-9
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Discriminator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.hidden_size=256
        self.lstm1 = nn.LSTM(input_size, self.hidden_size, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dout = nn.Dropout(0.2)
        self.l2 = nn.Linear(self.hidden_size, 64)
        self.l3 = nn.Linear(64, 1)
        self.l4 = nn.Linear(3000, 1)

    def forward(self, x, alpha):
        x  = ReverseLayerF.apply(x, alpha)
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to('cuda')
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to('cuda')
        x, (_, _) = self.lstm1(x, (h_0, c_0))
        x = x.view(-1, self.hidden_size)
        x = F.relu(self.l1(x)) # lstm, linear, relu, dropout 0.3, linear, relu, linear, lieanr, sigmoid
        x = self.dout(x)
        x = self.l3(F.relu(self.l2(x)))
        x = torch.sigmoid(self.l4(x.view(-1)))
        return x

class CoDATS_Discriminator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 128),
            # nn.BatchNorm1d(128), # batch size =1 이라 없앰
            nn.ReLU(),
            nn.Dropout(0.3))
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3))
        self.l1 = nn.Linear(128, 1)


    def forward(self, x, alpha):
        x = torch.mean(x.view(x.size(0), 128, -1), dim=2) # global avg pooling
        reversed_x = ReverseLayerF.apply(x, alpha)
        x= self.block1(reversed_x)
        x = self.block2(x)
        x = torch.sigmoid(self.l1(x).view(-1))
        return x

class CoDATS_task_classifier(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.l1(x)
        x = torch.sigmoid(x) # no softmax for cross entropy loss
        return x


class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()
        #in_channels: input의 feature dimension
        #out_channels: 내가 output으로 내고싶은 dimension
        self.output_size = output_size
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(5, 128, kernel_size=8,stride=1, bias=False, padding= 'same'), # #Out = (3000+2P-8)=2999
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, bias=False, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, self.output_size, kernel_size=8, stride=1, bias=False,padding='same'),
            nn.BatchNorm1d(self.output_size),
            nn.ReLU())

        # self.AvgPool1d = nn.AvgPool1d(5) #(same,same,features_len)

    def forward(self, x_in):
        x = x_in.reshape(1,5,-1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x = x[1,128,:-1]
        x = x.reshape(1,-1, self.output_size)
        #x_flat = x.reshape(x.shape[0], -1)
        return x

class Build_DANN(nn.Module):
    def __init__(self,backbone):
        super(Build_DANN, self).__init__()
        device = 'cuda'

        if backbone == 'CNN':
            self.feature_extractor = CNN(128).to(device)
            # self.task_classifier = CoDATS_task_classifier(133).to(device)
            self.domain_classifier = CoDATS_Discriminator(128).to(device)
            self.kinematics = Supervision_kinematics(133, 256, device).to(device)
        else:
            # self.task_classifier = Supervision_collision(10, 128, device).to(device)
            self.domain_classifier = Discriminator(5).to(device)
            self.kinematics = Supervision_kinematics(10, 256, device).to(device)
            if backbone == 'VRNN':
                self.feature_extractor = VRNN().to(device)
            elif backbone == 'LSTM':
                self.feature_extractor = Encoder(5, 256).to(device)



def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size 4

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).unsqueeze(0).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c


def CORAL(source, target):
    d = source.size(1)  # dim vector ## 4, 5

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


class Build_Discrepancy(nn.Module):
    def __init__(self,backbone):
        super(Build_Discrepancy, self).__init__()
        device = 'cuda'
        self.task_classifier = Supervision_collision(10, 128, device).to(device)
        self.kinematics = Supervision_kinematics(10, 256, device).to(device)
        if backbone == 'CORAL':
            self.feature_extractor = CNN(5).to(device)
        elif backbone == 'LSTM_CORAL' or 'LSTM_MMD':
            self.feature_extractor = Encoder(5, 256).to(device)
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class DSN(nn.Module):
    def __init__(self):
        super(DSN, self).__init__()
        device = 'cuda'
        self.kinematics = Supervision_kinematics(10, 256, device).to(device)
        self.private_target_encoder = Encoder(5, 256).to(device)
        self.private_source_encoder = Encoder(5, 256).to(device)
        self.shared_encoder = Encoder(5, 256).to(device)
        self.shared_decoder = Decoder(10,5, 256).to(device)

def fix_randomness(SEED, train):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if train:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
