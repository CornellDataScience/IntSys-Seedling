
# coding: utf-8

# In[25]:


import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import Conv2d, ReLU, MaxPool2d
from torch.autograd import Variable
from torchvision import models
import torchvision.models as models
import copy
import pickle
import cv2
import sys
import numpy as np
import time
from torchsummary import summary
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import os
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

#loading data 1

name = "VGG_PRUNE"
if not os.path.exists(name):
    os.mkdir(name)

print(name)

# In[3]:


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

data_dir = os.path.join('.', 'data')

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

image_dataset = datasets.ImageFolder(data_dir, transform=data_transform)
dataset_size = len(image_dataset)
class_names = image_dataset.classes

# calc balanced count
class_counts = {}

for i in range(len(image_dataset)):
    label_index = image_dataset[i][1]
    class_counts[label_index] = class_counts.get(label_index, 0) + 1

balanced_count = None
balanced_class = None
for class_ in class_counts:
    if balanced_class is None or class_counts[class_] < class_counts[balanced_class]:
        balanced_class = class_
        balanced_count = class_counts[balanced_class]
print("balanced count finished")

dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=32,
        shuffle=True, num_workers=4)

# Get a batch of training data
inputs, classes = next(iter(dataloader))
with open('test2_indices.pickle', 'rb') as file:
    test_indices = pickle.load(file)

with open('train2_indices.pickle', 'rb') as file:
    train_indices = pickle.load(file)

print("images loaded")
vgg = torch.load('/home/jek343/IntSys-Seedling/FINAL_VGG/VGG11_FINAL_MODEL.pt') #load my model here


# In[36]:


def freeze_layers(model, n):
    i = 0
    for name, param in model.named_parameters():
        if i < n:
            param.requires_grad = False
        else:
            break
        i += 1


# In[37]:


num_features = vgg.classifier[6].in_features
features = list(vgg.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg.classifier = nn.Sequential(*features) # Replace the model classifier

n = 0
for param in vgg.parameters():
    n += 1
n_unfrozen = 1
frozen = n - n_unfrozen * 2
freeze_layers(vgg, frozen)

lr = 0.001
momentum = 0.9
step_size = 7
gamma = 0.1

print('lr', lr)
print('momentum', momentum)
print('step_size', step_size)
print('gamma', gamma)

optimizer_ft = optim.SGD(vgg.parameters(), lr=lr, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)


# In[16]:


# loading data 3
def indicesSplit(ds, balanced_size, percent_train=0.9):
    train_indices = []
    test_indices = []
    counts = {}

    for i in range(len(ds)):
        label_index = ds[i][1]

        counts[label_index] = counts.get(label_index, 0) + 1

        if counts[label_index] < balanced_size * percent_train:
            train_indices.append(i)

        elif counts[label_index] < balanced_size:
            test_indices.append(i)


    return train_indices, test_indices

print("indices split finished")

# loading data 4
k = int(252*.9)

train_indices, test_indices = indicesSplit(image_dataset, balanced_count)

train_ds = Subset(image_dataset, train_indices)
test_ds = Subset(image_dataset, test_indices)
train_dataloader = DataLoader(
        train_ds, batch_size=32,
        shuffle=True, num_workers=4)

test_dataloader = DataLoader(
        test_ds, batch_size=32,
        shuffle=True, num_workers=4
    )

print("dataloader finished")

def train_model(model, criterion, optimizer, scheduler, train_dataloader, test_dataloader, num_epochs=10):
    if torch.cuda.is_available():
        model.cuda() #.cuda() will move everything to the GPU side
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    train_batches = len(train_dataloader)
    val_batches = len(test_dataloader)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        n_train = 0
        n_val = 0

        model.train(True)

        for i, data in enumerate(train_dataloader):
            print("\rTraining batch {}/{}".format(i + 1, train_batches), end='', flush=True)


            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            n_train += len(preds)
            loss_train += loss.data.item()
            acc_train += torch.sum(preds == labels.data).item()

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_train = loss_train / n_train
        avg_acc_train = acc_train / n_train
        train_losses.append(avg_loss_train)
        train_accs.append(avg_acc_train)
        model.train(False)
        #model.eval()

        for i, data in enumerate(test_dataloader):
            print("\rValidation batch {}/{}".format(i+1, val_batches), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            n_val += len(preds)
            loss_val += loss.data.item()
            acc_val += torch.sum(preds == labels.data).item()

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / n_val
        avg_acc_val = acc_val / n_val
        val_losses.append(avg_loss_val)
        val_accs.append(avg_acc_val)


        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss_train))
        print("Avg acc (train): {:.4f}".format(avg_acc_train))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, train_losses, train_accs, val_losses, val_accs

def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    n = 0

    test_batches = len(test_dataloader)
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(test_dataloader):
        print("\rTest batch {}/{}".format(i+1, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        n += len(preds)
        loss_test += loss.data.item()
        acc_test += torch.sum(preds == labels).item()

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / n
    avg_acc = acc_test / n

    elapsed_time = time.time() - since
    return elapsed_time,avg_loss,avg_acc

def evaluate(model):
    if torch.cuda.is_available():
        model.cuda() #.cuda() will move everything to the GPU side
    criterion = nn.CrossEntropyLoss()
    return eval_model(model, criterion)

def create_dict(model):
    print(model)
    num_layers = len(model.features._modules)
    conv_dict = {}
    for i in range(num_layers):
        if (type(model.features._modules[str(i)]))==torch.nn.modules.conv.Conv2d:
            conv = model.features._modules[str(i)]
            num_filters = model.features._modules[str(i)].in_channels
            for filter_index in range(num_filters):
                weights = conv.weight.data.cpu().numpy()[i][filter_index]
                conv_dict[(i,filter_index)] = np.mean(np.abs(weights))
    conv_dict = OrderedDict(sorted(conv_dict.items(), key=lambda x: x[1]))
    conv_dict = list(conv_dict.keys())[:9]
    return conv_dict

# In[20]:


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


# In[103]:


def prune_conv_layer(model, layer_index, filter_index):
    conv = model.features._modules[str(layer_index)]
    next_conv = None
    offset = 1
    print("\nPRUNING LAYER " + str(layer_index) + " FILTER " + str(filter_index))
    print("WEIGHT DATA")
    print(conv.weight.data.cpu().numpy()[layer_index][filter_index])
    while layer_index + offset <  len(model.features._modules.items()):
        res =  model.features._modules[str(layer_index+offset)]
        if isinstance(res, torch.nn.modules.conv.Conv2d):
            next_name = str(layer_index+offset) 
            next_conv = res
            break
        offset = offset + 1
    
    new_conv =         torch.nn.Conv2d(in_channels = conv.in_channels,             out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = True)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
    if torch.cuda.is_available():
        new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    else:
        new_conv.weight.data = torch.from_numpy(new_weights).cpu()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
    if torch.cuda.is_available():
        new_conv.bias.data = torch.from_numpy(bias).cuda()
    else:
        new_conv.bias.data = torch.from_numpy(bias).cpu()

    if not next_conv is None:
        next_new_conv =             torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,                out_channels =  next_conv.out_channels,                 kernel_size = next_conv.kernel_size,                 stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation,
                groups = next_conv.groups,
                bias = True)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
        if torch.cuda.is_available():
            next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()
        else:
            next_new_conv.weight.data = torch.from_numpy(new_weights).cpu()

        next_new_conv.bias.data = next_conv.bias.data

    if not next_conv is None:
        features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
                    [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
        del model.features
        del conv

        model.features = features

    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        model.features = torch.nn.Sequential(
                *(replace_layers(model.features, i, [layer_index], \
                    [new_conv]) for i, _ in enumerate(model.features)))
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1

        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        params_per_input_channel = int(old_linear_layer.in_features / conv.out_channels)
        
        new_linear_layer =             torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features)

        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()

        new_weights[:, : filter_index * params_per_input_channel] =             old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] =             old_weights[:, (filter_index + 1) * params_per_input_channel :]

        new_linear_layer.bias.data = old_linear_layer.bias.data
        
        if torch.cuda.is_available():
            new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()
        else:
            new_linear_layer.weight.data = torch.from_numpy(new_weights).cpu()

        classifier = torch.nn.Sequential(
            *(replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))

        del model.classifier
        del next_conv
        del conv
        model.classifier = classifier
        print(model.features)
    return model


# In[107]:
def prune(model, acc):
    pretrain_epoch = 50
    avg_acc = 1
    while avg_acc > 0.8:
        conv_dict = create_dict(model)
        for i in conv_dict:
            model = prune_conv_layer(model, i[0], i[1])
        criterion = nn.CrossEntropyLoss()
        model, train_losses, train_accs, val_losses, val_accs = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, train_dataloader, test_dataloader, num_epochs=pretrain_epoch)
        torch.save(model, name + '/model.pt')
        np.save(name + "/train_losses", np.array(train_losses))
        np.save(name + "/train_accs", np.array(train_accs))
        np.save(name + "/val_losses", np.array(val_losses))
        np.save(name + "/val_accs", np.array(val_accs))
        _,avg_loss,avg_acc = evaluate(model)
        acc.append(avg_acc)
    return model, acc

our_vgg_model, acc = prune(vgg, []) # change to reflect what conv layer we want to prune
torch.save(our_vgg_model, '/home/jek343/pruned_and_trained')
print (" The accuracy after pruning is " + str(acc))
epochs = range(len(acc))
print(epochs)
print(acc)
