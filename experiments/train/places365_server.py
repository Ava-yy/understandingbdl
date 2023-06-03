import os
import flask
from flask import Flask, request
from flask_cors import CORS

import numpy as np
import sys
import json
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from base64 import b64encode

from sklearn import mixture


base_dir_0 = '/home/zhaoy32/Desktop/understandingbdl/experiments/train/ckpts/'

dataset_name = 'places365_3c'

dataset_dir = 'places365_multiswag_3c_resnet50'

base_dir = os.path.join(base_dir_0,dataset_dir,'multiswag')

image_fn_list = json.load(open(os.path.join(base_dir,dataset_name+'_val_fn_list.json'),'r'))

multiswag_prediction = json.load(open(os.path.join(base_dir,'multiswag_predictions.json'),'r'))

if dataset_name == 'places365_3c':
    class_list = ['classroom','conference_room','supermarket']
else:
    class_list =['candy_store','classroom','coffee_shop','computer_room','conference_center', 'conference_room', 'lecture_room', 'office', 'supermarket', 'toyshop']

# ({'predictions': [10000,100],'targets':[10000]})

IMG_SIZE = (224, 224) 
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])



app = Flask(__name__)
CORS(app)

# @app.route('/calculate_jacobian', methods=['GET','POST'])

# def calculate_jacobian():

#     client_data = flask.request.json
#     image_id = client_data['image_id']
#     sample_id = client_data['sample_id']
#     class_id = client_data['class_id']
#     sample_model_dir = os.path.join(la_sample_model_dir,'la_googlenet_fc_sample_'+str(sample_id)+'.pt')
#     checkpoint_fc = torch.load(sample_model_dir)
#     # vgg16_model.load_state_dict(checkpoint["state_dict"])
#     model_ft.fc.load_state_dict(checkpoint_fc) # we saved the state_dict of each sample googlenet model as .pt  
#     model_ft.eval()

#     image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
#     image = Image.open(image_path)
#     image = test_transform(image)
#     x = image.cuda(non_blocking=True).unsqueeze(0)
#     x.requires_grad_(True)
#     y = model_ft(x) # shape (1,10) tensor([[ -3.0459,  -9.2013,  -8.5121,   3.3400, -12.8477,  -6.3842,  -3.1321, -3.0611,  -0.2655,  -3.6536]], device='cuda:0', grad_fn=<AddmmBackward0>)
#     predict_id = torch.argmax(y[0])
#     y_class = y[0][class_id] #(1,10)
#     if x.grad is not None:
#         x.grad.data.fill_(0)       
#     y_class.backward(retain_graph=True) #(1,3,224,224)
#     print('grad_x shape :',x.grad.data.shape) #(1,3,224,224)
#     jacobian = x.grad.data[0].cpu().data.numpy()
#     print('jacobian shape : ',jacobian.shape) #(3,224,224)
 
#     return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'sample_id':sample_id,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})    

@app.route('/gmm_3c', methods=['GET','POST'])

def gmm_3c():

    client_data = flask.request.json
   
    image_id = client_data['image_id']

    n_components = client_data['n_components']

    coord_data = client_data['coord_data'] # all pixel coord including pixels inside of the triangle

    img_data = np.load(open(os.path.join(base_dir,'image_data/image_'+str(image_id)+'_data.npy'),'rb'))

    # print('img_data.shape: ',img_data.shape)

    img_data = img_data.reshape((-1,img_data.shape[-1]))

    print('img_data shape : ',img_data.shape)

    print('img_data sum : ',np.sum(img_data))
    # img_data = np.random.rand(60,3)

    # img_data = img_data/img_data.sum(axis=0)

    gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)

    gmm_data = gmm_model.fit(img_data)

    coord_data = np.array(coord_data)

    coord_data = coord_data.reshape((-1,coord_data.shape[-1]))

    predictions = gmm_data.predict(coord_data)

    predictions_prob = gmm_data.predict_proba(coord_data)

    score_samples = gmm_data.score_samples(coord_data)

    return flask.jsonify({
        'data':img_data.tolist(),
        'mean': gmm_data.means_.tolist() ,
        'covariance': gmm_data.covariances_.tolist(),
        'weights': gmm_data.weights_.tolist(),
        'predictions': predictions.tolist(), 
        'predictions_proba': predictions_prob.tolist(),
        'converged': gmm_data.converged_,
        'AIC':gmm_data.aic(img_data),
        'BIC':gmm_data.bic(img_data),
        'score_samples':score_samples.tolist()})
 

# @app.route('/gmm', methods=['GET','POST'])

# def gmm():

#     client_data = flask.request.json
   
#     image_id = client_data['image_id']

#     n_components = client_data['n_components']

#     img_data = np.load(open(os.path.join(base_dir_0,dataset_dir,'multiswag/image_data/image_'+str(image_id)+'_data.npy'),'rb'))

#     img_data = img_data.reshape((-1,img_data.shape[-1]))

#     gmm_model = mixture.GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)

#     gmm_data = gmm_model.fit(img_data)

#     predictions = gmm_data.predict(img_data)

#     predictions_prob = gmm_data.predict_proba(img_data)

#     score_samples =gmm_data.score_samples(img_data)

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
  

@app.route('/image_data', methods=['GET','POST'])

def image_data():

    client_data = flask.request.json
   
    image_id = client_data['image_id']
   
    img_data = np.load(open(os.path.join(base_dir,'image_data/image_'+str(image_id)+'_data.npy'),'rb'))

    return flask.jsonify(img_data.tolist())


@app.route('/image_multiswag_pred', methods=['GET','POST'])

def image_multiswag_pred():

    client_data = flask.request.json
   
    image_id = client_data['image_id']

    res = multiswag_prediction['predictions'][image_id]
   
    return flask.jsonify(res)


@app.route('/image_data_3c', methods=['GET','POST'])

def image_data_3c():

    client_data = flask.request.json
   
    image_id = client_data['image_id']

    img_data = np.load(open(os.path.join('/home/zhaoy32/Desktop/understandingbdl/experiments/train/ckpts/places365_multiswag_3c_resnet50/multiswag/image_data/image_'+str(image_id)+'_data.npy'),'rb'))

    img_data = img_data.reshape((-1,img_data.shape[-1]))
   
    return flask.jsonify(img_data.tolist())
 


    
@app.route('/image_path', methods=['GET','POST'])

def image_path():

    client_data = flask.request.json
   
    image_id = client_data['image_id']

    return flask.jsonify({'image_path':image_fn_list[image_id][0]  ,'image_label':image_fn_list[image_id][1],'image_label_name':class_list[int(image_fn_list[image_id][1])] })
 

# @app.route('/calculate_jacobian_finetune', methods=['GET','POST'])

# def calculate_jacobian_finetune():

#     client_data = flask.request.json
#     image_id = client_data['image_id']
#     class_id = client_data['class_id']
    
#     image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
#     image = Image.open(image_path)
#     image = test_transform(image)
#     x = image.cuda(non_blocking=True).unsqueeze(0)
#     x.requires_grad_(True)
#     y = model_ft(x) # shape (1,10) tensor([[ -3.0459,  -9.2013,  -8.5121,   3.3400, -12.8477,  -6.3842,  -3.1321, -3.0611,  -0.2655,  -3.6536]], device='cuda:0', grad_fn=<AddmmBackward0>)
#     predict_id = torch.argmax(y[0])
#     y_class = y[0][class_id] #(1,10)
#     if x.grad is not None:
#         x.grad.data.fill_(0)       
#     y_class.backward(retain_graph=True) #(1,3,224,224)
#     print('grad_x shape :',x.grad.data.shape) #(1,3,224,224)
#     jacobian = x.grad.data[0].cpu().data.numpy()
#     print('jacobian shape : ',jacobian.shape) #(3,224,224)
 
#     return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})
    

# # @app.route('/calculate_jacobian', methods=['GET','POST'])

# # def calculate_jacobian():

# #     client_data = flask.request.json
# #     image_id = client_data['image_id']
# #     sample_id = client_data['sample_id']
# #     class_id = client_data['class_id']
# #     sample_model_dir = os.path.join(la_sample_model_dir,'la_googlenet_sample_'+str(sample_id)+'.pt')
# #     checkpoint = torch.load(sample_model_dir)
# #     # vgg16_model.load_state_dict(checkpoint["state_dict"])
# #     model_ft.load_state_dict(checkpoint) # we saved the state_dict of each sample googlenet model as .pt  
# #     model_ft.eval()
# #     image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
# #     image = Image.open(image_path)
# #     image = test_transform(image)
# #     x = Variable(image,requires_grad=True)
# #     x = x.cuda(non_blocking=True).unsqueeze(0)
# #     y = model_ft(x)
# #     predict_id = torch.argmax(y[0])
# #     y = y[0][class_id] #(1,10)
# #     grad_x, = torch.autograd.grad(y, x, create_graph=True)
# #     jacobian = grad_x.reshape(x.shape).data[0].cpu().data.numpy() #(1,3,224,224)
# #     print('x shape : ',x.shape)
# #     return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'sample_id':sample_id,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})
   


# @app.route('/image_la_kl', methods=['GET','POST'])

# def get_image_la_kl():
#     # client_data = flask.request.json
#     # image_id = client_data['image_id']
#     data = json.load(open(base_dir+'image_la_kl.json','r'))
#     return flask.jsonify({'data':data})


# @app.route('/image_la_result', methods=['GET','POST'])

# def get_image_la_result():
#     client_data = flask.request.json
#     image_id = client_data['image_id']
#     data = json.load(open(image_la_result_dir+'image_'+str(image_id)+'.json','r'))
#     return flask.jsonify({'image_id':image_id,'data':data})



# @app.route('/la_all_images_result', methods=['GET','POST'])

# def get_la_all_images_result():
#     client_data = flask.request.json
#     sample_id = client_data['sample_id']
#     data = json.load(open(os.path.join(la_all_images_result_dir,'la_test_sample_'+str(sample_id)+'.json'),'r')) # for googlenet data
#     # data = json.load(open(la_all_images_result_dir+'result_dict_'+str(sample_id)+'.json','r')) #for resnet and vgg data
#     return flask.jsonify({'sample_id':sample_id,'data':data})
    


# @app.route('/la_all_images_result_bma', methods=['GET','POST'])

# def get_la_all_images_result_bma():
#     #client_data = flask.request.json

#     res = []
#     for sample_id in range(100):

#         data = json.load(open(os.path.join(la_all_images_result_dir,'la_test_sample_'+str(sample_id)+'.json'),'r')) # for googlenet data
#         res.append(data)
#     # data = json.load(open(la_all_images_result_dir+'result_dict_'+str(sample_id)+'.json','r')) #for resnet and vgg data
#     return flask.jsonify({'data':res})


# @app.route('/calculate_jacobian_avg', methods=['GET','POST'])

# def calculate_jacobian_avg():

#     client_data = flask.request.json
#     image_id = client_data['image_id']
#     data = json.load(open(os.path.join(base_dir,'jac_avg','jac_avg_'+str(image_id)+'.json'),'r'))
#     for i in range(len(data)): # class id
#         data[i] = {'label_id': i, 'label': CHOOSED_CLASSES[i] ,'data': data[i]}
#     return flask.jsonify({'jacobian_avg':data,'image_id':image_id})
    


if __name__ == '__main__':
    app.run()





