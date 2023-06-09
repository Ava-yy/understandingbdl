import itertools
import torch
import os
import json
import copy
from datetime import datetime
import math
import numpy as np
import tqdm
from collections import defaultdict
from time import gmtime, strftime
import sys

import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

def get_logging_print(fname):
    cur_time = strftime("%m-%d_%H:%M:%S", gmtime())

    def print_func(*args):
        str_to_write = ' '.join(map(str, args))
        filename = fname % cur_time if '%s' in fname else fname
        with open(filename, 'a') as f:
            f.write(str_to_write + '\n')
            f.flush()

        print(str_to_write)
        sys.stdout.flush()

    return print_func, fname % cur_time if '%s' in fname else fname


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i+n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=False, verbose=False, subset=None,
                regularizer=None):
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)
        if regularizer:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f' % (
                verb_stage + 1, loss_sum / num_objects_current,
                correct / num_objects_current * 100.0
            ))
            verb_stage += 1

    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': None if regression else correct / num_objects_current * 100.0,
        'stats': {key: value / num_objects_current for key, value in stats_sum.items()}
    }


def eval(loader, model, criterion, cuda=True, regression=False, verbose=False, eval=True):
    loss_sum = 0.0
    correct = 0.0
    stats_sum = defaultdict(float)
    num_objects_total = len(loader.dataset)

    model.train(not eval)
    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output, stats = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)
            for key, value in stats.items():
                stats_sum[key] += value

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / num_objects_total,
        'accuracy': None if regression else correct / num_objects_total * 100.0,
        'stats': {key: value / num_objects_total for key, value in stats_sum.items()}
    }


# denormalize an image tensor after normalization
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

# # tensor_x: size [B, 3, H, W]
# torchvison.utils.save_image(denormalize(tensor_x))


def denormalize_single(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1)

# # tensor_x: size [B, 3, H, W]
# torchvison.utils.save_image(denormalize(tensor_x))


def predict(loader, model, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0

    with torch.no_grad():

        for input, target in loader:  # loader len = 79

            input = input.cuda(non_blocking=True)  # (128,3,32,32)
            output = model(input)

            batch_size = input.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def predict_eval_single(model, image_path, eval_image_path, verbose=False):

    predictions = list()
    targets = list()

    model.eval()
    image_id = 0
    image_label = []
    print('eval image path : ', eval_image_path)

    IMG_SIZE = (224, 224) 
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    with torch.no_grad():

        # input = input.cuda(non_blocking=True) # (128,3,32,32)
        image = Image.open(image_path)

        image_transformed = test_transform(image).unsqueeze(0)
        img = image_transformed.to('cuda')  # (128,3,32,32)
        output = model(img)

        prediction = F.softmax(output, dim=1).cpu().numpy()

        #save_image(denormalize_single(img), os.path.join(eval_image_path,image_path))

    return {
        'prediction': predictions,
    }


def predict_eval(loader, model, eval_image_path, verbose=False):
    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    image_id = 0

    image_label = []

    print('eval image path : ', eval_image_path)

    with torch.no_grad():

        for image, target in loader:  # loader len = 79

            # input = input.cuda(non_blocking=True) # (128,3,32,32)
            image = image.to('cuda')  # (128,3,32,32)
            output = model(image)
            # output = model.base(image)

            batch_size = image.size(0)
            predictions.append(F.softmax(output, dim=1).cpu().numpy())
            targets.append(target.numpy())
            offset += batch_size

            print('start eval image')
            if not os.path.exists(eval_image_path):

                print('save image')
                for img_id, img in enumerate(image):
                    save_image(denormalize_single(img),
                               eval_image_path+'/img_'+str(image_id)+'.jpg')
                    image_id += 1

    # json.dump(image_label, open(eval_image_path+'/image_label.json', 'w'))
    # json.dump(eval_img_prediction,open(eval_image_path+'/eval_img_prediction.json','w'))

    return {
        'predictions': np.vstack(predictions),
        'targets': np.concatenate(targets)
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x/(1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def set_weights(model, vector, device=None):
    offset = 0
    for param in model.parameters():
        param.data.copy_(
            vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()


def extract_parameters(model):
    params = []
    for module in model.modules():
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            param = module._parameters[name]
            params.append((module, name, param.size()))
            module._parameters.pop(name)
    return params


def set_weights_old(params, w, device):
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape).to(device))
        offset += size


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


def accuracy(outputs, labels):
    return (np.argmax(outputs, axis=1) == labels).mean()


def calibration_curve(outputs, labels, num_bins=20):
    confidences = np.max(outputs, 1)
    step = (confidences.shape[0] + num_bins - 1) // num_bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(outputs, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    # ece = Variable(torch.zeros(1)).type_as(confidences)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin-accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    out = {
        'confidence': xs,
        'accuracy': ys,
        'p': zs,
        'ece': ece,
    }
    return out


def ece(outputs, labels, num_bins=20):
    return calibration_curve(outputs, labels, num_bins=20)['ece']