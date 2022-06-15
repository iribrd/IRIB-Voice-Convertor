import os
import pickle
import librosa
import torch
import numpy as np
from math import ceil
from model_vc import Generator
from synthesis import build_model
from synthesis import wavegen


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
#dim_neck,,,freq
G = Generator(64,256,512,16).eval().to(device)

g_checkpoint = torch.load('model_farsdat.ckpt',map_location=device)
G.load_state_dict(g_checkpoint['state_dict'])

metadata = pickle.load(open('test_metadata.pkl', "rb"))

spect_vc = []
rootDir = './test_spmel'

for sbmt_i in metadata:
    print('speaker id:')
    print(sbmt_i[0])
    print('speaker encoding:')
    print(sbmt_i[1])
    print("content encoding :")
    print(sbmt_i[2])
    print("**** x_org ****")
    x_org = np.load(os.path.join(rootDir,  sbmt_i[2]))
    #x_org = np.load(sbmt_i[2])    
    print(x_org)
    x_org, len_pad = pad_seq(x_org)
    print("**** x_org after padding**** ")
    print(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            #x_identic_psnt = x_identic_psnt.squeeze(0)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[ 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[ 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
with open('test_metadata_result21.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle) 
    
    
spect_vc = pickle.load(open('test_metadata_result21.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    librosa.output.write_wav(name + '.wav', waveform, sr=16000)