# from iCaRL import iCaRLmodel
# from iCaRL_nofc import iCaRLmodel
from iCaRL_online_nofc import iCaRLmodel
from ResNet import resnet18_cbam
from common import ResNet18, Custom_CLIP
import torch

numclass=10
# feature_extractor=resnet18_cbam()
# feature_extractor=ResNet18()
feature_extractor = Custom_CLIP()
img_size=32
batch_size=128
task_size=10
memory_size=2000
epochs=70
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

for i in range(10):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
