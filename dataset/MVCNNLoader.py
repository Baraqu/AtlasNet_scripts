import numpy as np
import torch
# import torch.optim as optim
# import torch.nn as nn
import os,shutil,json,glob
import argparse
from torch.autograd import Variable

# from tools.Trainer import ModelNetTrainer
# from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

from PIL import Image
from torchvision import transforms

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

clf = PCA(n_components=2)

np.random.seed(ord('c') + 137)

N = 1
n_texture = 11
num_of_views = 12
colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm']
markers = ['x', '^', 'o', 's']
shapes = ['Cube', 'Cylinder', 'Pyramid', 'Cone', 'Tetrahedron', 'Octahedron', 'Icosahedron', 'Dodecahedron', 'Pipe', 'Prism', 'Torus', 'Helix']
class MVCNN_Loader(data.Dataset):
    def __init__(self, opt):  
        # load config from local
        with open('mvcnn/config.json') as f:
            args = json.loads(f.read())
        
        pretraining = not args['no_pretraining']
        project_name = args['name']
        CNN_name = args['cnn_name']
        num_of_views = args['num_views']
        
        # set up model parameter
        cnet = SVCNN(project_name, nclasses=40, pretraining=pretraining, cnn_name=CNN_name)
        self.cnet_2 = MVCNN(project_name, cnet, nclasses=40, cnn_name=CNN_name, num_views=num_of_views)
        self.cnet_2.cuda()
        del cnet
        
       
    
    
    def get_feature_vectors(self, files, n=N):
        vectors= []
        for i in range(n):
            rand_idx = np.random.permutation(len(files))
            files = [files[rand_idx[i]] for i in range(num_of_views)]
            
            imgs = []
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])
            for file in files:
                # img = Image.open(file).convert('RGB').resize((224, 224))
                # print(np.asarray(img).shape)
                img = np.asarray(Image.open(file).convert('L').resize((224, 224)))
                img = np.expand_dims(img, axis=-1)
                # print(img.shape)
                im = transform(np.concatenate([img, img, img], axis=-1))
                imgs.append(im)
            
            data = torch.stack(imgs)
            # 'mvcnn'
            V,C,H,W = data.size()
            in_data = Variable(data).view(-1,C,H,W).cuda()
                
            out_data = cnet_2.forward1(in_data)
                # for key in out_data.keys():
                    # print(key, out_data[key].shape)
            feature_vector = out_data['5'].data.cpu().numpy()
            vectors.append(feature_vector)
        return vectors

    def MVCNN_IMG_Loader(self):
        # load config from local
        with open('mvcnn/config.json') as f:
            args = json.loads(f.read())
        
        pretraining = not args['no_pretraining']
        project_name = args['name']
        CNN_name = args['cnn_name']
        num_of_views = args['num_views']
        
        # set up model parameter
        cnet = SVCNN(project_name, nclasses=40, pretraining=pretraining, cnn_name=CNN_name)
        cnet_2 = MVCNN(project_name, cnet, nclasses=40, cnn_name=CNN_name, num_views=num_of_views)
        cnet_2.cuda()
        del cnet
        
        cnet_2.load('./mvcnn_stage_2')
        
        # set phase to be 'Test'
        cnet_2.eval()
        
        # out_data = None
        # in_data = None
        
        vectors = []
        
        for shape in shapes:
            for i in range(n_texture):
                files = np.array(glob.glob('K:/T800/{:s}/Images/*'.format(shape + str(i))))
                temp_vectors = get_feature_vectors(files)
                vectors.extend(temp_vectors)
        
        feature_vectors = np.concatenate(vectors, axis=0)
        print('[INFO] feature_vectors.shape:', feature_vectors.shape)
        
        return feature_vectors
   