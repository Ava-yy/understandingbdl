import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import json

from swag import data, models, utils, losses
from swag.posteriors import SWAG_single
#from swag.posteriors import SWAG
from swag import data_places365_10c


savedir = 'ckpts/places365_multiswag_10c_300samples/multiswag'

image_label_prediction = json.load(open(os.path.join(savedir,'multiswag_predictions.json'),'r')) # {'predictions':multiswag_probs.tolist(),'targets':targets.tolist()}

image_label = json.load(open(os.path.join(savedir,'places365_10c_val_fn_list.json'),'r'))

res = []

def kl_divergence(a, b):
    return sum(a[i] * np.log(a[i]/b[i]) for i in range(len(a)))

nnl = nn.NLLLoss()


for img_id in range(len(image_label_prediction['targets'])):

    data_prediction = np.load(open(os.path.join(savedir,'image_data','image_'+str(img_id)+'_data.npy'),'rb'))

    data_prediction = data_prediction.reshape((-1,data_prediction.shape[-1]))

    avg_prediction = np.mean(data_prediction,axis=0)

    bma_prediction = np.array(image_label_prediction['predictions'][img_id]) #(10,)

    # target = float(image_label[img_id][1]) 

    target = image_label_prediction['targets'][img_id]

    bma_nnl = nnl(torch.tensor([bma_prediction]),torch.tensor([target]))

    bma_nnl = float(bma_nnl)

    ekl = 0

    for sample_pred in data_prediction:

    	sample_kl = kl_divergence(avg_prediction,sample_pred) 

    	ekl += sample_kl

    ekl /= data_prediction.shape[0]

    print('ekl',ekl)


    res.append({'image_id':img_id,'image_label':target,'bma_prediction':avg_prediction.tolist(),'nnl':bma_nnl,'ekl': ekl})

json.dump(res,open(os.path.join(savedir,'acc_ekl_image.json'),'w'))









