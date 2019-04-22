#!/usr/bin/env python
# coding: utf-8

# # [helpful link](https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch)

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchsummary
from torchsummary import summary
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import time
import copy


# In[2]:


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")


# In[3]:


#loading data 1

data_dir = os.path.join('.', 'data')

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

image_dataset = datasets.ImageFolder(data_dir, transform=data_transform)

dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=32,
        shuffle=True, num_workers=4)

dataset_size = len(image_dataset)

print("Loaded images under {}".format(dataset_size))
    
print("Classes: ")
class_names = image_dataset.classes
print(image_dataset.classes)


# In[4]:


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
print(balanced_count)


# In[5]:


# loading data 2

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# Get a batch of training data
inputs, classes = next(iter(dataloader))
show_databatch(inputs, classes)


# In[6]:


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


# In[7]:


# loading data 4
k = int(252*.9)

train_indices, test_indices = indicesSplit(image_dataset, balanced_count)


# In[8]:


# loading data 5

train_ds = Subset(image_dataset, train_indices)
test_ds = Subset(image_dataset, test_indices)
train_dataloader = DataLoader(
        train_ds, batch_size=32,
        shuffle=True, num_workers=4
    )
test_dataloader = DataLoader(
        test_ds, batch_size=32,
        shuffle=True, num_workers=4
    )


# In[9]:


vgg16 = models.vgg16(pretrained=True)


# In[10]:


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


# In[11]:


freeze_layers(vgg16)
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier


# In[12]:


summary(vgg16, (3, 224, 224))


# In[13]:


def visualize_model(vgg, num_images=6):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        size = inputs.size()[0]
        
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training) # Revert model back to original training state


# In[14]:


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

        if use_gpu:
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
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


# In[15]:


if use_gpu:
    vgg16.cuda() #.cuda() will move everything to the GPU side
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[16]:


print("Test before training")
eval_model(vgg16, criterion)


# In[20]:


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
#     avg_loss = 0
#     avg_acc = 0
#     avg_loss_val = 0
#     avg_acc_val = 0
    
    
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
        
        vgg.train(True)
        
        for i, data in enumerate(train_dataloader):
            print("\rTraining batch {}/{}".format(i + 1, train_batches), end='', flush=True)
                
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            n_train += len(preds)
            loss_train += loss.data.item()
            acc_train += torch.sum(preds == labels.data).item()
#             print("p", preds)
#             print("l", labels.data)
#             print("\nn_correct", torch.sum(preds == labels.data).item())
#             print("n", len(preds))
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_train = loss_train / n_train
        avg_acc_train = acc_train / n_train
        
        vgg.train(False)
        #vgg.eval()
            
        for i, data in enumerate(test_dataloader):
            print("\rValidation batch {}/{}".format(i+1, val_batches), end='', flush=True)
                
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            n_val += len(preds)
            loss_val += loss.data.item()
            acc_val += torch.sum(preds == labels.data).item()
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / n_val
        avg_acc_val = acc_val / n_val
        
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
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    return vgg


# In[21]:


vgg16_trained = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')


# In[19]:


eval_model(vgg16_trained, criterion)


# In[20]:


print(len(train_dataloader))
print(len(test_dataloader))


# In[22]:


len(train_indices)


# In[23]:


len(test_indices)


# In[24]:


2827/32


# In[28]:


n_val


# In[ ]:




