import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from torch.autograd import Variable
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torchvision
import pickle
import os
import numpy as np

import xlwt
from xlwt import Workbook


#some constants
CAT_CNT = 12
TRAIN_SIZE = 2732
VALID_SIZE = 304


def inter_forward(model, x):
    mod_list = list(model.modules())
    for l in mod_list[:len(mod_list)-1]:
        x = l(x)
    return l


def create_model(frozenLayers):

    #load a pretrained resnet model
    res = torchvision.models.resnet34(pretrained=True)

    #freezing the first forzenLayers layers
    ct = 0
    for name, child in res.named_children():
       ct += 1
       if ct < frozenLayers:
           for name2, params in child.named_parameters():
               params.requires_grad = False

    #counting in-features for fully connected layer
    n_inputs = res.fc.in_features

    #create fully connected layer with 12 out features + activation layer + softmax
    res.fc = nn.Sequential(nn.Linear(n_inputs, 128),
                          nn.LeakyReLU(),
                          nn.BatchNorm1d(128),
                          nn.Linear(128, CAT_CNT),
                          nn.BatchNorm1d(CAT_CNT),
                          nn.LeakyReLU(),
                          nn.LogSoftmax(dim = 1))
    return res

def create_dataloaders(BATCH_SIZE):
    #unpickling the data files
    #files are trainX_128, trainY_128, validX_128, validY_128
    data_path = os.path.join(".", "balanced_pickled")
    trainX = pickle.load(open(os.path.join(data_path, "trainX_128" ), "rb"))
    trainY = pickle.load(open(os.path.join(data_path, "trainY_128" ), "rb"))
    validX = pickle.load(open(os.path.join(data_path, "validX_128" ), "rb"))
    validY = pickle.load(open(os.path.join(data_path, "validY_128" ), "rb"))
    #print(len(trainX))
    #print(len(validX))
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


def train_model(model, BATCH_SIZE, paramlr, optimlr, epochsNum, start, sheet1, m):
    running_loss = 0
    running_corrects = 0
    phase = 'train'
    model.train()
    criterion = nn.CrossEntropyLoss()

    #optimizer creation
    param_groups = [

    {'params':model.fc.parameters(),'lr': paramlr},
    ]
    optimizer = optim.Adam(param_groups, lr=optimlr)


    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()

    (t_loader, v_loader) = create_dataloaders(BATCH_SIZE)

    epochs = epochsNum
    #steps = 0
    #train_losses, test_losses = [], []
    sheet1.write(0, 9+m, 'BS:' + str(BATCH_SIZE) + ',PLR:' + str(paramlr) + ',OLR:' + str(optimlr))

    for epoch in range (epochs):
        print('Epoch{}/{}.'.format(epoch, epochs))
        print('-' * 10)

        running_loss = 0.0
        running_vloss = 0.0
        running_corrects = 0.0
        running_valid_corrects = 0.0

        tl = next(iter(t_loader))
        for i, (inputs, labels) in enumerate(tqdm(tl)):
            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            #clears the gradients of all optimized tensors
            optimizer.zero_grad()

            #forwards + backwards + optimize
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            #print(preds.double())

            #print statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == torch.max(labels, 1)[1])
            #print(torch.sum(preds == torch.max(labels, 1)[1]))

            #print('ok;')
        epoch_loss = running_loss/(TRAIN_SIZE)
        epoch_acc = running_corrects.double() / (TRAIN_SIZE)


        if epoch%1 == 0:
            model.eval()

            vl = next(iter(v_loader))
            cnt = 0
            for j, (vinputs, vlabels) in enumerate(vl):
                cnt+=1
                if torch.cuda.is_available():
                    vinputs= vinputs.to(device)
                    vlabels = vlabels.to(device)

                vloss = criterion(outputs, torch.max(labels, 1)[1])

                #intermediate = inter_forward(model, vinputs)
                #print(intermediate)

                #forwards
                voutputs = model.forward(vinputs)

                _, vpreds = torch.max(voutputs, 1)
                running_valid_corrects += torch.sum(vpreds == torch.max(vlabels, 1)[1])

                running_vloss += vloss.item()
                #print(running_valid_corrects)
                #print('ok;')
            valid_loss = running_vloss/(VALID_SIZE)
            valid_acc = running_valid_corrects.double() / (VALID_SIZE)
            #print(cnt)
            sheet1.write(epoch + start - 1, 0, 'BS:' + str(BATCH_SIZE) + ',PLR:' + str(paramlr) + ',OLR:' + str(optimlr) + ', Epoch: ' + str(epoch))
            sheet1.write(epoch + start - 1, 1, epoch_loss)
            sheet1.write(epoch + start - 1, 2, epoch_acc.double().item())
            sheet1.write(epoch + start - 1, 3, valid_acc.double().item())
            sheet1.write(epoch + start - 1, 4, valid_loss)
            sheet1.write(epoch + start - 1, 5, epoch)
            sheet1.write(epoch + 1, 9+m, valid_acc.double().item())
            print('{} Loss: {:.4f} Acc: {:.4f} Valid Acc: {:.4f} Valid Loss: {:.4f}'.format(phase, epoch_loss, epoch_acc.double(), valid_acc, valid_loss))
        print()


    #testing model on specific classes
    nb_classes = CAT_CNT
#    confusion_matrix = torch.zeros(nb_classes, nb_classes)
#    with torch.no_grad():
#        for i, (inputs, classes) in enumerate(t_loader['val']):
#
#            if torch.cuda.is_available():
#                inputs = inputs.to(device)
#                classes = classes.to(device)
#            outputs = model(inputs)
#            _, preds = torch.max(outputs, 1)
#            for t, p in zip(classes.view(-1), preds.view(-1)):
#                    confusion_matrix[t.long(), p.long()] += 1
   # print(confusion_matrix)

    #print(confusion_matrix.diag()/confusion_matrix.sum(1))

    #torch model save
    torch.save(model, "full_model")
    torch.save(model.state_dict(), "modelRes.pt")

    #onnx model save
    dummy_input = torch.randn(BATCH_SIZE, 3, 128, 128)
    if torch.cuda.is_available():
        dummy_input = dummy_input.to(device)
    torch.onnx.export(model, dummy_input, "resNet_notPrune.onnx")
   # print(model.state_dict())

def main():
    model = create_model(15)
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    #model = 1
    print('test')
    #4 apart, 4 apart, 3 apart
    #lr1 = .0001
  
    #train_model(model, bs, plr, olr, epochs, startLine in excel, sheet1, model)
    #startline shouls always be >= 2
    #model should increase by 1

    train_model(model, 64, .0001, .00001, 5, 2, sheet1, 1)
    train_model(model, 32, .0001, .00001, 5, 8, sheet1, 2)
    #print('bad')
    sheet1.write(0, 0, 'Parameters and Epoch')
    sheet1.write(0, 1, 'Epoch Loss')
    sheet1.write(0, 2, 'Epoch Acc')
    sheet1.write(0, 3, 'Valid Acc')
    sheet1.write(0, 4, 'Valid Loss')
    wb.save('TuningHyperParams.xls')


if __name__ == "__main__":

    main()
