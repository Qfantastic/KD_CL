import torch.nn as nn
import torch
import random
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import clip

writer = SummaryWriter("CustomCLIP_experiment0402_online_fc_ep=30_every5batch")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_clip, preprocess = clip.load('ViT-B/32', device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate):

        super(iCaRLmodel, self).__init__()
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.target_set = []
        self.full_target_set = []
        self.feature_extractor_output_set = []
        self.numclass = numclass

        self.loss_l2 = nn.MSELoss()

  

        self.steps = 0
        self.len_indexs = []


        # self.transform = transforms.Compose([#transforms.Resize(img_size),
        #                                      transforms.ToTensor(),
        #                                     #  preprocess
        #                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                
        #                                     ])
        # self.old_model = None

        # self.train_transform = transforms.Compose([
        #                                           transforms.RandomCrop((32,32),padding=4),
        #                                           transforms.RandomHorizontalFlip(p=0.5),
        #                                           transforms.ColorJitter(brightness=0.24705882352941178),
        #                                           transforms.ToTensor(),
        #                                         #   preprocess
        #                                           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        #                                           ])
        
        # self.test_transform = transforms.Compose([
        #                                            transforms.ToTensor(),
        #                                         #    preprocess
        #                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        #                                          ])
        
        # self.classify_transform=transforms.Compose([
        #                                             transforms.RandomHorizontalFlip(p=1.),
        #                                             # #transforms.Resize(img_size),
        #                                             transforms.ToTensor(),
        #                                             # preprocess
        #                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        #                                            ])
        
        #************************

 
        self.transform = transforms.Compose([#transforms.ToTensor()
                                              preprocess
                                            ])
        self.old_model = None

        self.train_transform = transforms.Compose([#transforms.ToTensor()
                                                   preprocess
                                                  ])
        
        self.test_transform = transforms.Compose([#transforms.ToTensor()
                                                    preprocess
                                                 ])
        
        self.classify_transform=transforms.Compose([#transforms.ToTensor()
                                                     preprocess
                                                   ])
        
        self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
       # self.train_dataset_1 = iCIFAR100('dataset', transform=self.transform_, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size=memory_size
        self.real_memory_size = 0
        self.task_size=task_size

        self.train_loader=None
        self.test_loader=None
        self.train_loader_1=None

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        # images_test = self.train_dataset.get_image_class(0)
        # print("images_test.shape++++++++++++++",np.shape(images_test))
        classes=[self.numclass-self.task_size,self.numclass]
        self.train_loader,self.test_loader=self._get_train_and_test_dataloader(classes)
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        
        self.checkfull_of_classes()
        print("self.steps",self.steps)
        print("self.len_indexs",self.len_indexs)
        self.model.train()
        self.model.to(device)
        # print("beforetrain part exemplar set",np.shape(self.exemplar_set[0]))
        # print("beforetrain part exemplar set value",self.exemplar_set[0])

    def checkfull_of_classes(self):
        m=int(self.memory_size/self.numclass)
        self.real_memory_size = int(m*self.numclass)
        self.full_target_set = []
        for i in range(self.numclass):
            self.full_target_set.append(torch.full([m],i))
        self.full_target_set = torch.cat(self.full_target_set)
            

        # for step, (indexs, images, target) in enumerate(self.train_loader_1):
        #     #images, target = images.to(device), target.to(device)
        #     # print("indexs:*^^^^^^^^^^^^^^^^^^^^^^^^^",indexs)
        #     # print("target:$$$$$$$$$$$$$$$$$$$$",target)
        #     # print("target_size",np.shape(target))
        #     # print("step",step)
        #     if(step == 0):
        #         for i in range(self.numclass-self.task_size,self.numclass):
        #             index = (target == i).nonzero(as_tuple = True)[0]
        #             exemplar_ = images[index]
               
        #             # print("exemplar_.shape++++++++++++++",np.shape(exemplar_))
        #         # break

        # print("----------------------------------------------------")
        for step, (indexs, images, target) in enumerate(self.train_loader):
            #images, target = images.to(device), target.to(device)
            print("indexs:*^^^^^^^^^^^^^^^^^^^^^^^^^",indexs)
            print("target:$$$$$$$$$$$$$$$$$$$$",np.shape(target))

            if(step == 0):
                for i in range(self.numclass-self.task_size,self.numclass):
                    index = (target == i).nonzero(as_tuple = True)[0] 
                    exemplar_ = images[index]
                    len_i = len(index)
                    self.len_indexs.append(len_i)
                    self.exemplar_set.append(exemplar_)
                  
                    
                len_index_array = np.array(self.len_indexs)


                if(len(np.where(len_index_array == 0)[0]) == 0):
                    if(len(np.where(len_index_array > m)[0]) > 0):
                        self._reduce_exemplar_sets_check(m)
                    self.steps = step
                    self.target_set = self.exemplar_set.copy()
                    for i_class in range(len(self.exemplar_set)):
                        self.target_set[i_class] = torch.full([len(self.exemplar_set[i_class])],i_class)

                    
                    print("******************end in step 0**********************")
                    break
            if(step !=0):
                for i in range(self.numclass-self.task_size,self.numclass):
                    index = (target == i).nonzero(as_tuple = True)[0]
                    exemplar_ = images[index]
                    len_i = len(index)
                    self.len_indexs[i] = self.len_indexs[i] + len_i
                    self.exemplar_set[i] = torch.cat((self.exemplar_set[i],exemplar_),0)

                if(len(np.where(len_index == 0)[0]) == 0):
                    if(len(np.where(len_index > m)[0]) > 0):
                        self._reduce_exemplar_sets_check(m)
                    self.steps = step
                    self.target_set = self.exemplar_set.copy()
                    for i_class in range(len(self.exemplar_set)):
                        self.target_set[i_class] = torch.full([len(self.exemplar_set[i_class])],i_class)
                    print("******************end in step N**********************")
                    break
            



            

                
  



    def _get_train_and_test_dataloader(self, classes):
        #self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  pin_memory=True,
                                  num_workers=4)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batchsize,
                                 pin_memory=True,
                                 num_workers=4)

        return train_loader, test_loader
    
    '''
    def _get_old_model_output(self, dataloader):
        x = {}
        for step, (indexs, imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                old_model_output = torch.sigmoid(self.old_model(imgs))
            for i in range(len(indexs)):
                x[indexs[i].item()] = old_model_output[i].cpu().numpy()
        return x
    '''

    # train model
    # compute loss
    # evaluate model
    def train(self):
        self.model.eval()
        print("train part exemplar set",np.shape(self.exemplar_set[0]))
        self.compute_exemplar_class_mean()
        self.model.train()
        accuracy = 0
        m=int(self.memory_size/self.numclass)
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        for epoch in range(self.epochs):
            # if epoch == 48:
            #     if self.numclass==self.task_size:
            #          print(1)
            #          opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
            #     else:
            #          for p in opt.param_groups:
            #              p['lr'] =self.learning_rate/ 5
            #          #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #     print("change learning rate:%.3f" % (self.learning_rate / 5))
            # elif epoch == 62:
            #     if self.numclass>self.task_size:
            #          for p in opt.param_groups:
            #              p['lr'] =self.learning_rate/ 25
            #          #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #     else:
            #          opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
            #     print("change learning rate:%.3f" % (self.learning_rate / 25))
            # elif epoch == 80:
            #       if self.numclass==self.task_size:
            #          opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
            #       else:
            #          for p in opt.param_groups:
            #              p['lr'] =self.learning_rate/ 125
            #          #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
            #       print("change learning rate:%.3f" % (self.learning_rate / 100))

            if epoch == 20:
                for p in opt.param_groups:
                    p['lr'] =self.learning_rate/ 5
                
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 23:
                for p in opt.param_groups:
                    p['lr'] =self.learning_rate/ 25
   
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 26:
                for p in opt.param_groups:
                    p['lr'] =self.learning_rate/ 125
   
                print("change learning rate:%.3f" % (self.learning_rate / 125))
           
            for step, (indexs, images, target) in enumerate(self.train_loader):

                if (step > self.steps):

                    #images, target = images.to(device), target.to(device)
                    #print("now the step is :",step)
                    memory_images, memory_target = self.sample_exemplar_class(sample_size = 64)    #sample size = 64
                   # memory_images, memory_target = memory_images.to(device), memory_target.to(device)
                    # print("memory_images shape:",np.shape(memory_images))
                    # print("memory_images shape:",np.shape(memory_images))
                    all_images = torch.cat((images,memory_images),0)
                    all_target = torch.cat((target,memory_target),0)

                    all_images, all_target = all_images.to(device), all_target.to(device)


                    #output = self.model(images)
                    #loss_value = self._compute_feature_loss_(indexs, all_images, all_target)   # stop #########################
                    loss_value = self._compute_loss(indexs, all_images, all_target)
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
                    #images, target = images.to(device), target.to(device)
                    self.update_exemplar_set(step, images, target)

                if(step%3==2):
                    print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            self._reduce_exemplar_sets_check(m)
            accuracy = self._test(self.test_loader, 0)
            print('epoch:%d,accuracy:%.3f,loss:%.3f' % (epoch, accuracy,loss_value.item()))
            writer.add_scalar('Accuracy/train:%d' % (self.numclass), accuracy, epoch)
            writer.add_scalar('Loss/train:%d' % (self.numclass), loss_value.item(), epoch)




        return accuracy

    def sample_exemplar_class(self, sample_size):
        example_all = torch.cat(self.exemplar_set)
        indices = torch.randperm(len(example_all))[:sample_size]
        self.target_set = self.exemplar_set.copy()
        for i_class in range(len(self.exemplar_set)):
            self.target_set[i_class] = torch.full([len(self.exemplar_set[i_class])],i_class)
        target_all = torch.cat(self.target_set)
        # if(len(example_all) < self.real_memory_size):
        #     self.target_set = self.exemplar_set.copy()
        #     for i_class in range(len(self.exemplar_set)):
        #         self.target_set[i_class] = torch.full([len(self.exemplar_set[i_class])],i_class)
        #     target_all = torch.cat(self.target_set)
        
        # else:
        #     target_all = self.full_target_set.clone().detach()


        return example_all[indices], target_all[indices]



    def update_exemplar_set(self, step, images, target):
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        for i in range(self.numclass-self.task_size,self.numclass):
            index = (target == i).nonzero(as_tuple = True)[0]
            exemplar_ = images[index]
            len_i = len(index)
            self.exemplar_set[i] = torch.cat((self.exemplar_set[i],exemplar_),0)
            self.len_indexs[i] = self.len_indexs[i] + len_i
        
        #len_index_array = np.array(self.len_indexs)
        if(step%5 == 0):
            self._reduce_exemplar_sets_check(m)
        self.model.train()




    def _test(self, testloader, mode):
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy


    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
       
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

    def _compute_feature_loss_(self, indexs, imgs, target):
        output = self.classify_(imgs)

        #output = get_one_hot(output, self.numclass)
        target = get_one_hot(target, self.numclass)
        
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            loss_classify = F.binary_cross_entropy_with_logits(output, target)
            print("loss_classify:",loss_classify)
            return loss_classify
        else:
            loss_classify = F.binary_cross_entropy_with_logits(output, target)
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_feature = self.old_model(imgs)
            new_feature = self.model(imgs)
            new_feature.requires_grad_(True)
            # loss_feature = torch.norm(new_feature-old_feature,2)
            loss_feature = self.loss_l2(new_feature,old_feature)
            # print("loss_classify:",loss_classify)
            # print("loss_feature:",loss_feature)
            L_all = loss_classify+loss_feature
            return L_all


    # change the size of examplar
    def afterTrain(self,accuracy):
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        writer.add_scalar('Size_exemplar', m, self.numclass)
        # self._reduce_exemplar_sets(m)
        # for i in range(self.numclass-self.task_size,self.numclass):
        #     print('construct class %s examplar:'%(i),end='')
        #     images=self.train_dataset.get_image_class(i)
        #     self._construct_exemplar_set(images,m)
        # self.numclass+=self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy=self._test(self.test_loader,0)
        writer.add_scalar('NMS_acc', KNN_accuracy.item(), self.numclass)
        self.numclass+=self.task_size
        print("NMS accuracy："+str(KNN_accuracy.item()))
        filename='model/accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, int(self.numclass-1) + 10)
        torch.save(self.model,filename)
        self.old_model=torch.load(filename)
        self.old_model.to(device)
        self.old_model.eval()
        


    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))

        print("feature_extractor_output:",np.shape(feature_extractor_output))
     
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        
        self.exemplar_set.append(exemplar)
        #self.exemplar_set.append(images)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def _reduce_exemplar_sets_check(self, m):
        self.compute_exemplar_class_mean()
        
        for index in range(len(self.exemplar_set)):
            
            if(len(self.exemplar_set[index]) > m):
                class_mean = self.class_mean_set[index]
                feature_extractor_output = self.feature_extractor_output_set[index]
                images = self.exemplar_set[index]
                examplar = []
                now_class_mean = np.zeros((1, 512))
                for i in range(m):
                    x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
                    x = np.linalg.norm(x, axis=1)
                    ind = np.argmin(x)
                    now_class_mean += feature_extractor_output[ind]
                    examplar.append(images[ind])

                self.exemplar_set[index] = torch.stack(examplar)
                self.len_indexs[index] = int(m)
                # print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))
        self.compute_exemplar_class_mean()




    def Image_transform(self, images, transform):

        # transform_ = transforms.Compose([
        #                               preprocess
        #                             ])
        # print("The images @@@@@@@@@@@@@@@@@@:",images[0].shape)
        #data = transform(Image.fromarray(images[0])).unsqueeze(0)
        data = transform_(Image.fromarray((images[0]*255).astype(np.uint8))).unsqueeze(0)
        for index in range(1, len(images)):
            #data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
            data = torch.cat((data, transform_(Image.fromarray((images[index]*255).astype(np.uint8))).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        #x = self.Image_transform(images, transform).to(device)

        x = images[0].unsqueeze(0)
        for index in range(1, len(images)):
            x = torch.cat((x, images[index].unsqueeze(0)), dim=0)
        #print("x**********************shape:",np.shape(x))
        
        # feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        x = x.to(device)
        feature_extractor_output = self.model.feature_extractor(x).detach().cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output


    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        self.feature_extractor_output_set = []
        #print("len of exemplar:",len(self.exemplar_set))

        for index in range(len(self.exemplar_set)):
            # print("compute the class mean of %s"%(str(index)))
            exemplar=self.exemplar_set[index]
            #exemplar=self.train_dataset.get_image_class(index)
            class_mean, feature_extractor_output = self.compute_class_mean(exemplar, self.transform)
            # class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            # class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            #class_mean = class_mean/np.linalg.norm(class_mean)
            self.class_mean_set.append(class_mean)
            self.feature_extractor_output_set.append(feature_extractor_output)

    def classify(self, test):
        result = []
        # test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)

    def classify_(self, test):
        result = []
        # test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        
        test = self.model(test).to(device)
        test.requires_grad_(True)
        class_mean_set = torch.tensor(np.array(self.class_mean_set)).to(device)
        for target in test:
            x = target - class_mean_set
            # x = torch.linalg.norm(x, ord=2, axis=1)
            # m = nn.Softmin(dim=0)
            # x = m(x)

            # x = torch.argmin(x)
            result.append(x)
        result = torch.stack(result)
        result = torch.linalg.norm(result, ord=2, axis=2)
        m = nn.Softmin(dim=1)
        result = m(result)

        return result

    # def softmax(self, x):
    # """Compute softmax values for each sets of scores in x."""
    #     e_x = np.exp(x - np.max(x))
    #     return e_x / e_x.sum() 
