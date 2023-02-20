import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.datasets as dst
from model import CLIP_CL
from data_loader import iCIFAR100
import argparse
import uuid
import torchvision.transforms as transforms
import clip

if __name__ == "__main__":


    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model parameters
    # parser.add_argument('--model', type=str, default='single',
    #                     help='model to train')
    # parser.add_argument('--n_hiddens', type=int, default=100,
    #                     help='number of hidden neurons at each layer')
    # parser.add_argument('--n_layers', type=int, default=2,
    #                     help='number of hidden layers')



    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
    # parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    # parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    # parser.add_argument('--cuda', type=int, default=1)

    # memory parameters
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--finetune', default='no', type=str,
                        help='whether to initialize nets in indep. nets')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    # parser.add_argument('--batch_size', type=int, default=10,
    #                     help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='yes',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--data_name', type=str, default='cifar100',help='name of dataset') # cifar10/cifar100

    parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')

    # Clip mode parameters

    parser.add_argument('--clip_backbone', type=str, default='ViT-B/32',
                    help='backbone model')


    
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False




    #**************** add the CLIP model


    # unique identifier
    uid = uuid.uuid4().hex


    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    K = 2000 # total number of exemplars

    clip_cl = CLIP_CL(512,1,args)
    # clip_cl = clip_cl
    



    if args.data_name == 'cifar10':
        total_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    elif args.data_name == 'cifar100':
        total_classes = 100
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)


    clip_model, preprocess = clip.load(args.clip_backbone, device=device)


    for s in range(0, total_classes, args.num_classes):
        # load Datasets

        print(f"Loading the training datasets {s} to {s+args.num_classes}...")

 




        train_set = iCIFAR100(root      = args.img_root,
                        train     = True,
                        classes = range(s,s+args.num_classes),
                        download  = True,
                        transform = preprocess)


        print(len(train_set))

        target = []

        for i in range(len(train_set)):
            if train_set.targets[i] not in target:
                target.append(train_set.targets[i])
        
        print("final target:",target)

        classes = list(set(train_set.targets))
        print("classes:",classes)
        
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size, shuffle=True)

        clip_cl.update_representation(train_set,args)

        m = K / clip_cl.n_classes

        #Reduce part : reduce the examplar sets for know classes
        clip_cl.reduce_exemplar_sets(m)

        # #Construct exemplar sets for new classes
        # for y in range(icarl.n_known, icarl.n_classes):
        #     print("Constructing exemplar set for class-%d..." %(y))
        #     images = train_set.get_image_class(y)
        #     clip_cl.construct_exemplar_set(images, m, transform_test)
        #     print("Done")
