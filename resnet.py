import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle
import os
import numpy as np

#some constants
CAT_CNT = 12
TRAIN_SIZE = 2732
VALID_SIZE = 304
BATCH_SIZE = 32

def create_model():
    
    #load a pretrained resnet model
    res = torchvision.models.resnet50(pretrained=True)
    
    #freeze model weights
    for param in res.parameters():
        param.requires_grad = False
    
    #counting in-features for fully connected layer
    n_inputs = res.fc.in_features
    
    #create fully connected layer with 12 out features + activation layer + softmax
    res.fc = nn.Sequential(nn.Linear(n_inputs, 512),
                          nn.ReLU(),
                          nn.Linear(512, CAT_CNT),
                          nn.ReLU(),
                          nn.LogSoftmax(dim = 1))
    
    return res
    

def create_dataloaders():
    #unpickling the data files
    #files are trainX_128, trainY_128, validX_128, validY_128
    data_path = os.path.join(".", "balanced_pickled")
    trainX = pickle.load(open(os.path.join(data_path, "trainX_128" ), "rb"))
    trainY = pickle.load(open(os.path.join(data_path, "trainY_128" ), "rb"))
    validX = pickle.load(open(os.path.join(data_path, "validX_128" ), "rb"))
    validY = pickle.load(open(os.path.join(data_path, "validY_128" ), "rb"))
    
    #generate data from the pickled np datasets,transforming to torch tensors
    trainX = np.transpose(trainX, (0,3,2,1))
    validX = np.transpose(validX, (0,3,2,1))
    
    tensor_trainX = Variable(torch.from_numpy(np.array(trainX)).float(), requires_grad=False)
    tensor_trainY = Variable(torch.from_numpy(np.array(trainY)).long(), requires_grad=False)
    train = TensorDataset(tensor_trainX, tensor_trainY)
    trainLoader = {DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)}

    tensor_validX = torch.stack([torch.Tensor(i) for i in validX])
    tensor_validY = torch.stack([torch.Tensor(i) for i in validY])
    valid = TensorDataset(tensor_validX, tensor_validY)
    validLoader = { DataLoader(valid)}
    
    return (trainLoader, validLoader)


def train_model(model):   
    running_loss = 0
    running_corrects = 0
    phase = 'train'
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()
    
    (t_loader, v_loader) = create_dataloaders()
    
    epochs = 3
    steps = 0
    #train_losses, test_losses = [], []
    
    for epoch in range (epochs):
        print('Epoch{}/{}.'.format(epoch, epochs-1))
        print('-' * 10)
        
        running_loss = 0.0
        
        
        tl = next(iter(t_loader))
        for i, (inputs, labels) in enumerate(tl):
            #inputs= inputs.to(device)
            #labels = labels.to(device)
            
            steps +=1
            
            #clears the gradients of all optimized tensors
            optimizer.zero_grad()
            
            #forwards + backwards + optimize
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            loss = criterion(preds, torch.max(labels, 1)[1])
            print('train-loss', loss.item(),end=' ')
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
            print(running_corrects/1)
            
            
        epoch_loss = running_loss/(TRAIN_SIZE)
        epoch_acc = running_corrects.double() / (TRAIN_SIZE)
        
        vl = next(iter(v_loader))
        for j, (inputs, labels) in enumerate(vl):
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc.double()))
            
        print()
    
    #save the model
    torch.save(model.state_dict(),".")
    
    # While iterating over the dataset do training
   
      
      
def main():
    model = create_model()
    train_model(model)
    
    
if __name__ == "__main__":
    main()