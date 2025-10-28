import torch
import threading
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  

from src.modelling.architecture import Archi, Archi1
from torchvision import transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt 

from io import BytesIO
import matplotlib.pyplot as plt
import sys, os
from src.modelling.config.basic import ConfigBasic
from src.modelling.networks.util import prepare_model
import numpy as np
#import torch
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Dataset
cfg = ConfigBasic()
cfg.dataset = 'aiims'
cfg.setting = 'D'
cfg.fold = 4

cfg.logscale = False
cfg.set_dataset()
cfg.tau = 1

# Model
cfg.model = 'GOL'
cfg.backbone = 'vgg16v2norm'
cfg.metric = 'L2'
cfg.k = np.arange(2, 50, 2)
cfg.epochs = 40
cfg.scheduler = 'cosine'
cfg.lr_decay_epochs = [100, 200, 300]
cfg.period = 3

cfg.margin = 0.25
cfg.ref_mode = 'flex'
cfg.ref_point_num = 10  # 60 Fold1, 58 Fold0 setting D // 56 setting c // 58 setting B // 55 setting A
cfg.drct_wieght = 1
cfg.start_norm = True
cfg.learning_rate = 0.0001

# Log
#cfg.wandb = False
#cfg.experiment_name = 'EXP_NAME'
#cfg.save_folder = f'../../RESULT_FOLDER_NAME/{cfg.dataset}/setting{cfg.setting}/{cfg.experiment_name}/PREFIX_{cfg.margin}_tau{cfg.tau}_F{cfg.fold}_{cfg.model}_{cfg.backbone}_{get_current_time()}'
#make_dir(cfg.save_folder)

cfg.n_gpu = torch.cuda.device_count()
cfg.num_workers = 1


class NeuralNetwork(nn.Module):
    def __init__(self, split, ind=None) -> None:
            super().__init__()
            self.ind = ind
            self.model = prepare_model(cfg)
            self.model.load_state_dict(torch.load("src/modelling/models/split_"+split+"/0/model.tar", map_location=torch.device('cpu')))
            if self.ind==0:
                 self.factor = -1
            else:
                 self.factor = 1
            
    
    def forward(self, x):
        x = self.model(x)
        #dist_mat = torch.matmul(x[0], self.model.ref_points[self.ind].T)
        dist_mat = torch.matmul(x, self.model.ref_points.T)
        dist_mat = dist_mat*self.factor
        #return torch.stack([dist_mat,])
        return dist_mat
    
class NeuralNetwork_pred(nn.Module):
    def __init__(self, split, ind=None) -> None:
            super().__init__()
            self.ind = ind
            self.model = prepare_model(cfg)
            self.model.load_state_dict(torch.load("src/modelling/models/split_"+split+"/0/model.tar", map_location=torch.device('cpu')))
            
    
    def forward(self, x):
        x = self.model(x)
        #dist_mat = torch.matmul(x[0], self.model.ref_points[self.ind].T)
        #dist_mat = torch.matmul(x, self.model.ref_points.T)
        #return torch.stack([dist_mat,])
        return x
    


import json
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import GuidedGradCam , GuidedBackprop, LRP, LayerGradCam, LayerAttribution

from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule
from captum.attr import visualization as viz
import io


import pickle
with (open("src/modelling/outputs_gol.pickle", "rb")) as openfile:
    outputs = pickle.load(openfile)
with (open("src/modelling/labels_gol.pickle", "rb")) as openfile:
    labels = pickle.load(openfile)

outputs = [i[0] for i in outputs]
labels = np.array(labels)

class Model():
    def __init__(self, start_callback=lambda: None, end_callback=lambda: None, progress_callback=lambda x: None):
        self._currentIndex = 0
        self.start_callback = start_callback
        self.end_callback = end_callback
        self.progress_callback = progress_callback
        self.start_callback()
        self.device = "cpu"
        #self.model = Archi().to(self.device)
        #self.model.load_state_dict(torch.load('src/parameters/models_overall/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        #self.model.eval()

        self.model1 = NeuralNetwork_pred("0", 1)
        #self.model1.load_state_dict(torch.load('src/parameters/models_param1/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        self.model1.eval()

        self.vis_model0 = NeuralNetwork(str(0), 0)
        self.vis_model0.eval()

        self.vis_model9 = NeuralNetwork(str(0), 9)
        self.vis_model9.eval()

        # self.model2 = Archi().to(self.device)
        # self.model2.load_state_dict(torch.load('src/parameters/models_param2/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        # self.model2.eval()

        # self.model3 = Archi().to(self.device)
        # self.model3.load_state_dict(torch.load('src/parameters/models_param3/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        # self.model3.eval()

        # self.model4 = Archi().to(self.device)
        # self.model4.load_state_dict(torch.load('src/parameters/models_param4/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        # self.model4.eval()

        # self.model5 = Archi().to(self.device)
        # self.model5.load_state_dict(torch.load('src/parameters/models_param5/ckpt.best.pth.tar', torch.device('cpu'))['state_dict'])
        # self.model5.eval()

        self.model6 = Archi1().to(self.device)
        self.model6.load_state_dict(torch.load('src/parameters/models_calculated/model_trained_ood.pth', torch.device('cpu'))['state_dict'])
        self.model6.eval()

        self.mt = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.results = [None] * 6
        self.threads = [None] * 6
        self.end_callback()

    @property
    def currentIndex(self):
        return self._currentIndex
    
    @currentIndex.setter
    def currentIndex(self, value):
        self._currentIndex = value
        self.progress_callback(self.currentIndex)

    def test(self, image, model):
        
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ])
        I = Image.fromarray(image)
        I = transform(I)
        
        with torch.no_grad():
            #output = model(torch.stack([I,]))
            pred = model(torch.stack([I,]))
            cs = torch.matmul(torch.stack(outputs), pred[0].T)
            _, inds = torch.topk(cs, 5)
            pred = round(sum(np.array(labels)[inds])/5)
            print("pred from model.test :" ,pred)

        return pred


    # def ood_test(self, image, model):
    #     transform = transforms.Compose([
    #             transforms.Resize((64, 64)),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ])
    #     image = Image.fromarray(image)
    #     image = transform(image)
    #     print(image.shape) 
        
    #     with torch.no_grad():
    #         output = model(torch.stack([image,]))
    #         print("Output from model.ood_test :",output)
    #         _, predicted=output.max(1)
    #         print("Predicted from model.ood_test :",predicted)
    #     return predicted
    

    # def override(self, x):
        
    #     models = [self.model1] 
    #     self.threads = [None] * len(models)
    #     self.results = [None] * len(models)
    #     self.currentIndex = 1
    #     self.progress_callback(self.currentIndex)
    #     pred = []
    #     for index, mdl in enumerate([self.model1],):# self.model2, self.model3, self.model4, self.model5, self.model]):
    #         self.threads[index] = threading.Thread(target = self.testing_func, args=(x, mdl, index), daemon=True)
    #         self.threads[index].start()
            
    #     for index, _ in enumerate(self.threads):
    #         self.threads[index].join()
    #         self.currentIndex = index + 2
    #         self.progress_callback(self.currentIndex)
    #     pred = self.results
    
    #     pred = [str(int(round(i))) for i in pred] 
    #     print("pred from model.override :",pred)
    #     return pred
    
    def testing_func(self, x, mdl, index):
        self.results[index] = self.test(x, mdl)
    
    def convert(self, img):
        img_buf = io.BytesIO()
        img.savefig(img_buf, format='png')
        return Image.open(img_buf)

    def get_concat_h(self, im1, im2):
        im1 = self.convert(im1)
        im2 = self.convert(im2)
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst


    def visualize(self, x):
        visualization = []
        for cls in [0,9]:
            if cls == 0:
                model = self.vis_model0
                sign = "negative"
            else:
                model = self.vis_model9
                sign = "positive"
            target_layer = model.model.encoder.features[-1]
            cap_vis = LayerGradCam(model, target_layer)
            

            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
            ])

            I = Image.fromarray(x)
            I = transform(I)
            
            gc_attr  = cap_vis.attribute(torch.stack([I,]), relu_attributions=False, target=cls )
            upsampled_gc_attr = LayerAttribution.interpolate(gc_attr,(224,224), interpolate_mode="bilinear")
    
            visualization.append(viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1,2,0).detach().numpy(),use_pyplot=True,original_image=x, method="blended_heat_map", sign=sign))
          
        viz_ = self.get_concat_h(visualization[0][0], visualization[1][0])

        return viz_


    def evaluate(self, x):
        
        self.threads = [None] * 1
        self.results = [None] * 1
        self.currentIndex = 0
        self.progress_callback(self.currentIndex)
        #val = self.ood_test(x,self.model6)
        #print(f'ood output : {val}') 
        self.currentIndex = 1
        self.progress_callback(self.currentIndex)
        # if val>0.0:
        #     pred = []
        #     for index, mdl in enumerate([self.model1,]):# self.model2, self.model3, self.model4, self.model5, self.model]):
        #         self.threads[index] = threading.Thread(target = self.testing_func, args=(x, mdl, index), daemon=True)
        #         self.threads[index].start()
        
        #     for index, _ in enumerate(self.threads):
        #         self.threads[index].join()
        #         self.currentIndex = index + 2
        #         self.progress_callback(self.currentIndex)
        #     pred = self.results
        #     print("pred from model.evaluate :",pred)
        #     self.currentIndex = 7
        #     self.progress_callback(self.currentIndex)
        #     #pred = [str(int(round(i))) for i in pred]
        #     #print(pred)
        #     return pred
        # else:
        #     self.currentIndex = 7
        #     self.progress_callback(self.currentIndex)
        #     return "Wrong Image!"

        
        pred = []
        for index, mdl in enumerate([self.model1,]):# self.model2, self.model3, self.model4, self.model5, self.model]):
            self.threads[index] = threading.Thread(target = self.testing_func, args=(x, mdl, index), daemon=True)
            self.threads[index].start()
    
        for index, _ in enumerate(self.threads):
            self.threads[index].join()
            self.currentIndex = index + 2
            self.progress_callback(self.currentIndex)
        pred = self.results
        print("pred from model.evaluate :",pred)
        self.currentIndex = 7
        self.progress_callback(self.currentIndex)
        return pred