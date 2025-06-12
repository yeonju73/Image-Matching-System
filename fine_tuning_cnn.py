# !pip install torchsummary

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from torchsummary import summary

import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

"""## Prepair Dataset"""

# !git clone https://github.com/hbcbh1999/recaptcha-dataset.git

# !rm -rf ./recaptcha-dataset/Large/Mountain/
# !rm -rf ./recaptcha-dataset/Large/Other/
# !rm -rf ./recaptcha-dataset/Large/readme.txt

"""ImageFolder structure

```
data_dir/Bicycle/xxx.png
data_dir/Bicycle/xxy.png
data_dir/Bicycle/[...]/xxz.png
...
data_dir/Traffic Light/123.png
data_dir/Traffic Light/nsdf3.png
data_dir/Traffic Light/[...]/asd932_.png
```


"""
def imshow(imgs, title=None):
    """Display image for Tensor."""
    imgs = imgs.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    imgs = std * imgs + mean
    imgs = np.clip(imgs, 0, 1)
    plt.imshow(imgs)
    if title is not None:
        plt.title(title)

"""## Build model

### ResNet from scratch

![resnet](https://pytorch.org/assets/images/resnet.png)
"""

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )



"""### Resnet from model zoo"""

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

"""## Train model"""

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




if __name__ == '__main__':
    data_dir = "./recaptcha-dataset/Large"
    class_names = ['Bicycle', 'Bridge', 'Bus', 'Car',
                'Chimney', 'Crosswalk', 'Hydrant',
                'Motorcycle', 'Palm', 'Traffic Light']

    input_size = 224
    batch_size = 32

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print("Initializing Datasets and Dataloaders...")

    image_datasets = datasets.ImageFolder(data_dir, data_transforms)  # your dataset
    num_data = len(image_datasets)
    indices = np.arange(num_data)
    np.random.shuffle(indices)

    train_size = int(num_data*0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_set = torch.utils.data.Subset(image_datasets, train_indices)
    val_set = torch.utils.data.Subset(image_datasets, val_indices)

    print('Number of training data:', len(train_set))
    print('Number of validation data:', len(val_set))

    dataloaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
                    'val': torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)}
    
    # Get a batch of training data
    inputs, labels = next(iter(dataloaders['train']))
    print("inputs.shape:", inputs.shape)
    print("labels.shape:", labels.shape)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs[:8])

    imshow(out, title=[class_names[x] for x in labels[:8]])

    # Get a batch of validation data
    inputs, labels = next(iter(dataloaders['val']))
    print("inputs.shape:", inputs.shape)
    print("labels.shape:", labels.shape)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs[:8])

    imshow(out, title=[class_names[x] for x in labels[:8]])
    
    model = ResNet_18(image_channels=3, num_classes=10)
    summary(model, (3, 224, 224), device='cpu')
    # summary(model, (3, 512, 512), device='cpu')
    
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    num_classes = 10
    num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    summary(model_ft, (3, 224, 224), device='cpu')
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

    """## Save the model & featuresl"""

    # 모델의 웨이트만 저장
    # torch.save(model_ft.state_dict(), 'resnet18.pt')

    torch.save(model_ft, 'resnet18_ft.pt')

    model_ft = torch.load('resnet18_ft.pt', weights_only=False)
    modules = list(model_ft.children())[:-1]
    resnet18_feat = nn.Sequential(*modules)

    for p in resnet18_feat.parameters():
        p.requires_grad = False
        
    # Test
    out = resnet18_feat(torch.rand(1, 3, 224, 224).to(device))
    print(out.shape)

    out = out.view([-1, 512])
    print(out.shape)

    out = out.detach().cpu().numpy()
    print(type(out))

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    for inputs, labels in tqdm(dataloaders['train']):
        inputs = inputs.to(device)
        h = resnet18_feat(inputs)

        # Eliminate unnecessary dimensions
        h = h.view([-1, 512])

        # Move to 'cpu' & change to 'numpy array'
        h = h.detach().cpu().numpy()

        train_features.append(h)

    # labels
    train_labels.append(labels.detach().cpu().numpy())

    for inputs, labels in tqdm(dataloaders['val']):
        inputs = inputs.to(device)
        h = resnet18_feat(inputs)

        # Eliminate unnecessary dimensions
        h = h.view([-1, 512])
        # Move to 'cpu' & change to 'numpy array'
        h = h.detach().cpu().numpy()

        val_features.append(h)

        # labels
        val_labels.append(labels.detach().cpu().numpy())
    
    train_features = np.concat(train_features, axis=0)
    train_labels = np.concat(train_labels, axis=0)
    val_features = np.concat(val_features, axis=0)
    val_labels = np.concat(val_labels, axis=0)

    print(f"Train Features: ({train_features.shape})")
    print(f"Train Labels: ({train_labels.shape})")
    print(f"Validation Features: ({val_features.shape})")
    print(f"Validation Labels: ({val_labels.shape})")
    
    """## KNN"""
    recaptcha = './recaptcha-dataset/Large/'
    labels = ['Bicycle','Bridge','Bus','Car','Chimney',
            'Crosswalk','Hydrant','Motorcycle','Palm','Traffic Light']

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_features, train_labels)

    predict_labels = classifier.predict(val_features)
    print(classification_report(val_labels, predict_labels, labels=labels))

    neigh_ind = classifier.kneighbors(X=val_features, n_neighbors=10, return_distance=False) # Top-10 results
    neigh_labels = np.array(train_labels)[neigh_ind]
    print(neigh_labels[:2])

    # 숫자를 이름으로 변경
    neigh_label_names = [[labels[idx] for idx in topk] for topk in neigh_labels]
    print(neigh_label_names[:2])

        
        
            
            
