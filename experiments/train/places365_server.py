import os
import flask
from flask import Flask, request
from flask_cors import CORS

import numpy as np
import sys
import json
from tqdm import tqdm

from PIL import Image
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from base64 import b64encode

from sklearn import mixture
from sklearn.cluster import KMeans

import argparse
import time
import tabulate

from swag import data, models, utils, losses
from swag.posteriors import SWAG_single
# from swag.posteriors import SWAG
from swag import data_places365_10c


base_dir_0 = '/home/zhaoy32/Desktop/understandingbdl/experiments/train/ckpts/'

dataset_name = 'places365_10c'
# dataset_name = 'places365_3c'

dataset_dir = 'places365_multiswag_10c_300samples'
# dataset_dir = 'places365_multiswag_3c'

base_dir = os.path.join(base_dir_0, dataset_dir, 'multiswag')

image_fn_list = json.load(
    open(os.path.join(base_dir, dataset_name+'_val_fn_list.json'), 'r'))

#multiswag_prediction = json.load(
#    open(os.path.join(base_dir, 'multiswag_predictions.json'), 'r'))

if dataset_name == 'places365_3c':
    class_list = ['classroom', 'conference_room', 'supermarket']
else:
    # class_list =['candy_store','classroom','coffee_shop','computer_room','conference_center', 'conference_room', 'lecture_room', 'office', 'supermarket', 'toyshop']
    class_list = ['banquet_hall', 'bar', 'beer_hall', 'cafeteria', 'coffee_shop',
                  'dining_hall', 'fastfood_restaurant', 'food_court', 'restaurant_patio', 'sushi_bar']

# CKPT_FILES = ['ckpts/places365_multiswag_10c/swag_1/swag-300.pt']# ckpts/places365_multiswag_10c/swag_2/swag-100.pt ckpts/places365_multiswag_10c/swag_3/swag-100.pt'

# parser = argparse.ArgumentParser(description='model sampling')
# parser.add_argument('--savedir', type=str, default='ckpts/places365_multiswag_10c/multiswag/', help='training directory (default: None)')
# parser.add_argument('--swag_ckpts', type=str, nargs='*', default=CKPT_FILES, help='list of SWAG checkpoints')
# parser.add_argument('--model_id', default='1', help="")
# parser.add_argument('--dataset', type=str, default='places365_10c', help='dataset name (default: CIFAR10)')
# parser.add_argument('--data_path', type=str, default='../../datasets/', metavar='PATH',
#                     help='path to datasets location (default: None)')
# parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
# parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size (default: 16)')
# parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
# parser.add_argument('--model', type=str, default='resnet50', metavar='model',
#                     help='model name (default: none)')
# parser.add_argument('--label_arr', default=None, help="shuffled label array")

# parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')

# parser.add_argument('--swag_samples', type=int, default=20, metavar='N',
#                     help='number of samples from each SWAG model (default: 20)')

# args = parser.parse_args()
# args.no_cov_mat = False

# if torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')
# # ({'predictions': [10000,100],'targets':[10000]})

# IMG_SIZE = (224, 224)
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]

# test_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(IMG_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(IMG_MEAN, IMG_STD)
# ])

app = Flask(__name__)
CORS(app)


# def sample_model_predictions(n_swag_samples,image_path): #image_data generation

#     loaders, num_classes,val_images_names_labels = data_places365_10c.loaders(
#         os.path.join(args.data_path, args.dataset.lower()),
#         args.batch_size,
#         args.num_workers,
#         # model_cfg.transform_train,
#         # model_cfg.transform_test,
#         shuffle_train=False)

#     os.makedirs(args.savedir+'n_samples'+str(n_swag_samples), exist_ok=True)
#     os.makedirs(args.savedir+'n_samples'+str(n_swag_samples)+'/eval_images',exist_ok=True)

#     model_class = getattr(torchvision.models, args.model)
#     print('model_class', model_class)
#     swag_model = SWAG_single(
#         model_class,
#         no_cov_mat=args.no_cov_mat,
#         # loading=True,
#         max_num_models=20,
#         num_classes=num_classes,
#     )
#     swag_model.to(args.device)

#     columns = ['swag', 'sample', 'te_loss', 'te_acc', 'ens_loss', 'ens_acc']

#     n_ensembled = 0.
#     multiswag_probs = None

#     total_predictions = []

#     for ckpt_i, ckpt in enumerate(args.swag_ckpts):

#         print("Checkpoint {}".format(ckpt))

#         checkpoint = torch.load(ckpt)

#         swag_model.load_state_dict(checkpoint['state_dict'])

#         swag_predictions = []

#         for sample in enumerate(range(n_swag_samples)):

#             swag_model.sample(.5)
#             utils.bn_update(loaders['train'], swag_model)

#             res = utils.predict_eval_single(
#                 model=swag_model,
#                 image_path=image_path,
#                 eval_image_path=args.savedir+'eval_images',
#                 verbose=False
#                 )

#             probs = res['prediction']
#             swag_predictions.append(probs) # (n_samples,n_images,n_classes)

#             # nll = utils.nll(probs, targets)
#             # acc = utils.accuracy(probs, targets)

#             # if multiswag_probs is None:
#             #     multiswag_probs = probs.copy()
#             # else:
#             #     #TODO: rewrite in a numerically stable way
#             #     multiswag_probs += (probs - multiswag_probs) / (n_ensembled + 1)
#             # n_ensembled += 1

#             # ens_nll = utils.nll(multiswag_probs, targets)
#             # ens_acc = utils.accuracy(multiswag_probs, targets)
#             # values = [ckpt_i, sample, nll, acc, ens_nll, ens_acc]
#             # table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
#             # print(table)

#         total_predictions.append(swag_predictions)

#     total_predictions = np.array(total_predictions) #(n_models,n_samples,n_images,n_classes)

#     print('total_predictions.shape : ',total_predictions.shape)

#     total_predictions = total_predictions.transpose(1,2,0,3)

#     return total_predictions[0] # (n_models,n_samples,n_classes)


# @app.route('/gmm_3c', methods=['GET','POST'])

# def gmm_3c():

#     client_data = flask.request.json

#     image_id = client_data['image_id']

#     n_components = client_data['n_components']

#     coord_data = client_data['coord_data'] # all pixel coord including pixels inside of the triangle

#     img_data = np.load(open(os.path.join(base_dir,'image_data/image_'+str(image_id)+'_data.npy'),'rb'))

#     # print('img_data.shape: ',img_data.shape)

#     img_data = img_data.reshape((-1,img_data.shape[-1]))

#     gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)

#     gmm_data = gmm_model.fit(img_data)

#     coord_data = np.array(coord_data)

#     coord_data = coord_data.reshape((-1,coord_data.shape[-1]))

#     predictions = gmm_data.predict(coord_data)

#     predictions_prob = gmm_data.predict_proba(coord_data)

#     score_samples = gmm_data.score_samples(coord_data)

#     return flask.jsonify({
#         'data':img_data.tolist(),
#         'mean': gmm_data.means_.tolist() ,
#         'covariance': gmm_data.covariances_.tolist(),
#         'weights': gmm_data.weights_.tolist(),
#         'predictions': predictions.tolist(),
#         'predictions_proba': predictions_prob.tolist(),
#         'converged': gmm_data.converged_,
#         'AIC':gmm_data.aic(img_data),
#         'BIC':gmm_data.bic(img_data),
#         'score_samples':score_samples.tolist()})


@app.route('/KMeans', methods=['GET', 'POST'])
def KMeans():

    client_data = flask.request.json

    image_id = client_data['image_id']
    n_components = client_data['n_components']
    sample_id_list = client_data['sample_id_list']

    # the softmax predictions from model samples
    img_data = np.load(open(os.path.join(base_dir_0, dataset_dir,
                       'multiswag/image_data/image_'+str(image_id)+'_data.npy'), 'rb'))
    img_data = img_data.reshape((-1, img_data.shape[-1]))

    img_data = img_data[np.array(sample_id_list)]

    # train a gmm based on the softmax predictions
    kmeans_data = KMeans(n_clusters=n_components,
                         random_state=0, n_init="auto").fit(img_data)

    return flask.jsonify({
        'data': kmeans_data.tolist(),
        'labels': kmeans_data.labels_.tolist(),
        'centers': kmeans_data.cluster_centers_.tolist(),
    })


@app.route('/gmm', methods=['GET', 'POST'])
def gmm():

    client_data = flask.request.json
    image_id = client_data['image_id']
    n_components = client_data['n_components']
    sample_id_list = client_data['sample_id_list']

    # the softmax predictions from model samples
    img_data = np.load(open(os.path.join(base_dir_0, dataset_dir,
                       'multiswag/image_data/image_'+str(image_id)+'_data.npy'), 'rb'))
    img_data = img_data.reshape((-1, img_data.shape[-1]))

    img_data = img_data[np.array(sample_id_list)]
    # train a gmm based on the softmax predictions
    gmm_model = mixture.GaussianMixture(
        n_components=n_components, covariance_type='full', random_state=0)

    gmm_data = gmm_model.fit(img_data)

    # predictions = gmm_data.predict(img_data)
    # predictions_prob = gmm_data.predict_proba(img_data)
    # score_samples =gmm_data.score_samples(img_data)

    return flask.jsonify({
        'data': img_data.tolist(),
        'mean': gmm_data.means_.tolist(),
        'covariance': gmm_data.covariances_.tolist(),
        'weights': gmm_data.weights_.tolist(),
        # 'predictions': predictions.tolist(),
        # 'predictions_proba': predictions_prob.tolist(),
        'converged': gmm_data.converged_,
        # 'AIC':gmm_data.aic(img_data),
        # 'BIC':gmm_data.bic(img_data),
        # 'score_samples':score_samples.tolist()
    })


@app.route('/gmm_on_the_fly', methods=['GET', 'POST'])
def gmm_on_the_fly():

    client_data = flask.request.json

    image_path = client_data['image_path']
    n_components = client_data['n_components']
    n_samples = client_data['n_samples']
    sample_id_list = client_data['sample_id_list']

    # the softmax predictions from model samples
    img_data = sample_model_predictions(n_samples, image_path)
    # img_data = np.load(open(os.path.join(base_dir_0,dataset_dir,'multiswag/image_data/image_'+str(image_id)+'_data.npy'),'rb'))

    img_data = img_data.reshape((-1, img_data.shape[-1]))

    img_data = img_data[np.array(sample_id_list)]

    # train a gmm based on the softmax predictions
    gmm_model = mixture.GaussianMixture(
        n_components=n_components, covariance_type='full', random_state=0)

    gmm_data = gmm_model.fit(img_data)

    # predictions = gmm_data.predict(img_data)
    # predictions_prob = gmm_data.predict_proba(img_data)
    # score_samples =gmm_data.score_samples(img_data)

    return flask.jsonify({
        'data': img_data.tolist(),
        'mean': gmm_data.means_.tolist(),
        'covariance': gmm_data.covariances_.tolist(),
        'weights': gmm_data.weights_.tolist(),
        # 'predictions': predictions.tolist(),
        # 'predictions_proba': predictions_prob.tolist(),
        'converged': gmm_data.converged_,
        # 'AIC':gmm_data.aic(img_data),
        # 'BIC':gmm_data.bic(img_data),
        # 'score_samples':score_samples.tolist()
    })


@app.route('/image_data', methods=['GET', 'POST'])
def image_data():

    client_data = flask.request.json

    image_id = client_data['image_id']

    sample_id_list = client_data['sample_id_list']

    img_data = np.load(open(os.path.join(
        base_dir, 'image_data/image_'+str(image_id)+'_data.npy'), 'rb'))

    return flask.jsonify(img_data[:, sample_id_list].tolist())


@app.route('/image_multiswag_pred', methods=['GET', 'POST'])
def image_multiswag_pred():

    client_data = flask.request.json

    image_id = client_data['image_id']

    res = multiswag_prediction['predictions'][image_id]

    return flask.jsonify(res)


@app.route('/image_data_3c', methods=['GET', 'POST'])
def image_data_3c():

    client_data = flask.request.json

    image_id = client_data['image_id']

    img_data = np.load(open(os.path.join(
        '/home/zhaoy32/Desktop/understandingbdl/experiments/train/ckpts/places365_multiswag_3c/multiswag/image_data/image_'+str(image_id)+'_data.npy'), 'rb'))

    img_data = img_data.reshape((-1, img_data.shape[-1]))

    return flask.jsonify(img_data.tolist())


@app.route('/image_path', methods=['GET', 'POST'])
def image_path():

    client_data = flask.request.json

    image_id = client_data['image_id']

    return flask.jsonify({'image_path': image_fn_list[image_id][0], 'image_label': image_fn_list[image_id][1], 'image_label_name': class_list[int(image_fn_list[image_id][1])]})


if __name__ == '__main__':
    app.run()
