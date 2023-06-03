import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

import json

from swag import data, models, utils, losses
from swag.posteriors import SWAG_single
#from swag.posteriors import SWAG
from swag import data_places365_10c

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--savedir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--swag_ckpts', type=str, nargs='*', required=True,
                    help='list of SWAG checkpoints')
parser.add_argument('--model_id', default='1', help="")

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size (default: 16)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='model',
                    help='model name (default: none)')
parser.add_argument('--label_arr', default=None, help="shuffled label array")

parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--swag_samples', type=int, default=20, metavar='N',
                    help='number of samples from each SWAG model (default: 20)')

args = parser.parse_args()
args.inference = 'low_rank_gaussian'
args.subspace = 'covariance'
args.no_cov_mat = False


if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
# args.device = 'cuda'


torch.backends.cudnn.benchmark = True

# model_cfg = getattr(models, args.model)

# print('Loading dataset %s from %s' % (args.dataset, args.data_path))
# loaders, num_classes = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     model_cfg.transform_train,
#     model_cfg.transform_test,
#     use_validation=not args.use_test,
#     split_classes=None
#     )

# places365 data loader
loaders, num_classes,val_images_names_labels = data_places365_10c.loaders(
    os.path.join(args.data_path, args.dataset.lower()), #args.data_path,
    args.batch_size,
    args.num_workers,
    # model_cfg.transform_train,
    # model_cfg.transform_test,
    shuffle_train=True)


if args.label_arr:
    print("Using labels from {}".format(args.label_arr))
    label_arr = np.load(args.label_arr)
    print("Corruption:", (loaders['train'].dataset.targets != label_arr).mean())
    loaders['train'].dataset.targets = label_arr

# print('Preparing model')
# model = model_cfg.base(*model_cfg.args, num_classes=num_classes,
#                        **model_cfg.kwargs)
# model.to(args.device)
# print("Model has {} parameters".format(sum([p.numel() for p in model.parameters()])))


model_class = getattr(torchvision.models, args.model)
print('model_class', model_class)

# model = torch.load(os.path.join("./", "places365_3c_resnet50_finetune.pt"))
# model.to(args.device)


# swag_model = SWAG(model_cfg.base,
#                 args.subspace, {'max_rank': args.max_num_models},
#                 *model_cfg.args, num_classes=num_classes,
#                 **model_cfg.kwargs)

swag_model = SWAG_single(
    model_class,
    no_cov_mat=args.no_cov_mat,
    # loading=True,
    max_num_models=20,
    num_classes=num_classes,
)
swag_model.to(args.device)


columns = ['swag', 'sample', 'te_loss', 'te_acc', 'ens_loss', 'ens_acc']

n_ensembled = 0.
multiswag_probs = None

print('Preparing directory %s' % args.savedir)

os.makedirs(args.savedir, exist_ok=True)
os.makedirs(args.savedir+'eval_images',exist_ok=True)

json.dump(val_images_names_labels,open(os.path.join(args.savedir,'places365_3c_val_fn_list.json'),'w'))

total_predictions = []

for ckpt_i, ckpt in enumerate(args.swag_ckpts):
    print("Checkpoint {}".format(ckpt))
    checkpoint = torch.load(ckpt)
    #swag_model.subspace.rank = torch.tensor(0)
    swag_model.load_state_dict(checkpoint['state_dict'])

    swag_predictions = []

    for sample in enumerate(range(args.swag_samples)):

        swag_model.sample(.5)
        utils.bn_update(loaders['train'], swag_model)

        swag_model.to('cpu')
        res = utils.predict_eval(
            loaders['test'],
            swag_model,
            eval_image_path=args.savedir+'eval_images',
            verbose=False
        )
        swag_model.to('cuda')

        probs = res['predictions']
        targets = res['targets']

        json.dump(targets.tolist(), open(os.path.join(args.savedir,'image_label.json'),'w'))

        swag_predictions.append(probs) # (n_samples,n_images,n_classes)

        nll = utils.nll(probs, targets)
        acc = utils.accuracy(probs, targets)

        if multiswag_probs is None:
            multiswag_probs = probs.copy()
        else:
            #TODO: rewrite in a numerically stable way
            multiswag_probs += (probs - multiswag_probs) / (n_ensembled + 1)
        n_ensembled += 1

        ens_nll = utils.nll(multiswag_probs, targets)
        ens_acc = utils.accuracy(multiswag_probs, targets)
        values = [ckpt_i, sample, nll, acc, ens_nll, ens_acc]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        print(table)

    # swag_predictions = np.array(swag_predictions) # (n_samples,n_images,n_classes)
    # for img_id, image_data in enumerate(swag_predictions.transpose(1, 0, 2) ):
    #     np.save(
    #         open(os.path.join(args.savedir, f'image_{img_id}_data_{args.model_id}.npy'),'wb'),
    #         image_data
    #     )

    total_predictions.append(swag_predictions)

total_predictions = np.array(total_predictions) #(n_models,n_samples,n_images,n_classes)

print('total_predictions.shape : ',total_predictions.shape)

total_predictions = total_predictions.transpose(1,2,0,3)

for img_id, image_data in enumerate(total_predictions.transpose(1,2,0,3)):

    np.save(open(os.path.join(args.savedir,'./image_'+str(img_id)+'_data.npy'),'wb'),image_data)


# print('Preparing directory %s' % args.savedir)
#os.makedirs(args.savedir, exist_ok=True)
with open(os.path.join(args.savedir, 'eval_command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

np.savez(os.path.join(args.savedir, "multiswag_probs.npz"),
         predictions=multiswag_probs,
         targets=targets)

json.dump({'predictions':multiswag_probs.tolist(),'targets':targets.tolist()},open(os.path.join(args.savedir,'./multiswag_predictions.json'),'w'))



