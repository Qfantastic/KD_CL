import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
import clip

from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/experiment_1')


device = "cuda" if torch.cuda.is_available() else "cpu"




class CLIP_CL(nn.Module):
    def __init__(self, feature_size,n_classes, args):
        super(CLIP_CL, self).__init__()
        self._backbone = args.clip_backbone
        self._device = device
        self.clip_model, self._preprocess = clip.load(args.clip_backbone, device=device)
        self.freeze()
        self.ReLU = nn.ReLU()

        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc  = nn.Linear(feature_size, n_classes, bias=False)

        self.n_classes = n_classes
        self.n_known = 0
        self.emb_size = 512



        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []
        self.classes_list = []

        # Learning method
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
        #                             weight_decay=0.00001)

        self.optimizer = torch.optim.SGD(self.parameters(),lr = args.lr)


        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

        
        if self._backbone == 'ViT-B/32' :
            self.emb_size = 512


        



        



    def freeze(self):
        '''
        Freeze all the layers to avoid training 
        '''
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode(self, images = None):
        '''
        Args : 
            image : images to clip encoded
        
        Returns :
            transformed image tensor
        '''

        encoded_images = images
        with torch.no_grad():
            features = self.clip_model.encode_image(encoded_images.to(self._device))
        
        self.train()

        return features


    def forward(self, x):


        x = self.clip_model.encode_image(x)
        x = self.bn(x)
        x = self.ReLU(x)


     

        # feat = x.to(torch.float32)
        # x = encode
        
        

        x = self.fc(x)


        return x

    def increment_classes(self,n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        # out_features = self.n_classes

        weight = self.fc.weight.data.cuda()
        print("in_features:",in_features)
        print("out_features:",out_features)

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        # self.fc = nn.Linear(in_features, self.n_classes, bias=False)
        self.fc.weight.data[:out_features] = weight
        print("self.fc************88",self.fc)
        # print("in_features:",in_features)
        # print("out_features:",out_features)
        # print("self.n_classes:",self.n_classes)
        self.n_classes += n


    def array_to_features(self,vx):
        bsize = vx.size(0)
        all_features = []
        for i in range(bsize):
            vx_i = vx[i]
            # vx_i = vx_i.reshape(3,32,32)
            vx_i = np.transpose(vx_i, (1,2,0))
            vx_i = Image.fromarray(np.uint8(vx_i*255),'RGB')
            vx_input = self._preprocess(vx_i).unsqueeze(0).to(self._device)
            # print("vx_input.shape++++++++++++++++++",vx_input.shape)
            image_features =  self.clip_model.encode_image(vx_input)
            all_features.append(image_features)
            
        # vx = vx.reshape(10,3,32,32)
        # print("vx.shape:",vx.shape)
        # # vx_out = np.transpose(vx, (0, 2, 3, 1))
        # features = model_clip.encode_image(vx.to(device))
        return torch.cat(all_features)


    def classify(self, x):
        """Classify images by neares-means-of-exemplars
        Args:
            x: input batch size
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print ("Done") 

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]


    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)



    def update_representation(self, dataset, args):

        self.compute_means = True
        # dataset = dataset.cuda()
        # Increment number of weights in final fc layer
        classes = list(set(dataset.targets))

        # new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        new_classes = [cls for cls in classes if cls not in self.classes_list]
        self.classes_list.extend(new_classes)
        # self.n_classes = len(self.classes_list)

        # print("self.classes_list:",self.classes_list)
        # print("self.n_classes:",self.n_classes)

        self.increment_classes(len(new_classes))

        

        print(f"{len(new_classes)} new classes")

        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)

        # print("dataset:",dataset)

        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                shuffle=True)
        

        #Store network outputs with pre-update parameters(pre_task old model)
        # q = torch.zeros(len(dataset), self.n_classes).cuda()
        # for indices, images, labels in loader:

        #     images_features = self.array_to_features(images)
        #     images_features = images_features.cuda()
        #     indices = indices.cuda()


        #     with torch.no_grad():
        #         feat, out = self.forward(images_features)
        #         g = torch.sigmoid(out)
        #         q[indices] = g.data
        # q = q.cuda()

        # network training
        optimizer = self.optimizer

        

        for epoch in range(args.n_epochs):

           
            for i, (indices, images, labels) in enumerate(loader):
                
            

                # image_features = self.clip_model.encode_image(images.to(self._device))

                # images_features = self.array_to_features(images)
                # print("images_features.shape:",np.shape(images_features))

                # images_features = images_features.to(self._device)
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()
                self.train()

                optimizer.zero_grad()

                # g = self.forward(images_features)

                
                out = self.forward(images)
                
                # Classification loss for new classes
                loss = self.cls_loss(out, labels)
                #loss = loss / len(range(self.n_known, self.n_classes))

                # # Distilation loss for old classes
                # if self.n_known > 0:
                    
                #     g = torch.sigmoid(out)
                #     q_i = q[indices]   #old model output
                #     dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y])\
                #             for y in range(self.n_known))
                #     #dist_loss = dist_loss / self.n_known
                #     loss += dist_loss

                loss.backward()
                optimizer.step()
            
                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                            %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0]))






class Norm_Net(nn.Module):
    def __init__(self,in_features,num_classes):
        super().__init__()

        # self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = x.to(torch.float32)
       
        # x = self.fc1(x)
        x = self.fc2(x)
        return x






def FC_Net2(in_features,nclasses):
    return Norm_Net(in_features,nclasses)