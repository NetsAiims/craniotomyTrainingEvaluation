import torch.nn as nn
from src.modelling.res2net import Resnet50
import torch.nn.init as init


class Archi(nn.Module):
    def __init__(self):
        super(Archi, self).__init__()
        self.model = Resnet50(embedding_size=128,
                     pretrained=True,
                     is_norm=1,
                     bn_freeze=1,
                     add_gmp=1
                     )
        #self.model = bn_inception(embedding_size=128,
        #                 pretrained=True,
        #                 is_norm=1,
        #                 bn_freeze=1,
        #                 add_gmp=1
        #                 )
        #path = "/home/raman/Work/Code/drilling/metric_learning/Proxy-Anchor-CVPR2020/logs/"+fld+"/logs_Drills_"+str(args.exp[-1])+"/resnet50_Proxy_Anchor_embedding128_alpha32_mrg0.1_adamw_lr0.0001_batch32/Drills_resnet50_best.pth"
        #path = "/home/raman/Work/Code/drilling/metric_learning/HIST/res/Drills/"+fld+"/run_"+str(args.exp[-1])+"/Drills_resnet50_best.pth"
        #self.model.load_state_dict(torch.load(path)['model_state_dict'])
        self.linear_relu_stack = nn.Sequential(
           # nn.Linear(512, 256),
            #nn.ReLU(),
            #nn.Linear(256, 128),
            #nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        #self.linear_relu_stack.apply(self._initialize_weights)

    def forward(self, x):
        x = self.model(x)
        logits = self.linear_relu_stack(x)
        return logits#, x
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out')
            init.constant_(m.bias, 0)
 
class Archi1(nn.Module):
    def __init__(self):
        super(Archi1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Assuming image size reduces to 6x6 after pooling
        self.fc2 = nn.Linear(128, 2)  # Two classes: in-distribution and OOD

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # print(x.shape)
        x = nn.MaxPool2d(2)(x)
        
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # x = nn.MaxPool2d(2)(x)
        # Use adaptive pooling here
        x = self.adaptive_pool(x)
        
        x = x.view(-1, 64 * 6 * 6)
        # x=nn.Flatten()(x)
        # print(x.shape)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        
        return nn.Softmax(dim=1)(x)