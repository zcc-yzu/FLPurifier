#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data_loader import DatasetForN
import math

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.trainloader, self.validloader, self.testloader = self.dataloader(
            dataset, idxs)
        if args.gpu == 'cuda':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def dataloader(self,dataset,idxs):
        trainloader = DataLoader(DatasetForN(dataset, idxs,10,False),
                                 batch_size=self.args.local_bs, shuffle=True)

        validloader = DataLoader(DatasetForN(dataset, idxs,10,False),
                                 batch_size=80, shuffle=False)
        testloader = DataLoader(DatasetForN(dataset, idxs,10,False),
                                batch_size=80, shuffle=False)
        return trainloader,validloader, testloader
    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        print(len(idxs))
        print(len(idxs_train))
        print(len(idxs_val))
        print(len(idxs_test))
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, preweight,global_round):
        # Set mode to train model
        # premodel=copy.deepcopy(model)
        # premodel.eval()
        #
        # for param in premodel.parameters():
        #     param.requires_grad = False

        model.train()
        epoch_loss = []
        epoch_loss1 = []
        epoch_loss2 = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                x1=images
                x2=images
                model.zero_grad()
                _, pre1, log_probs= model(x1)
                # _, pre2, _ = model(x2)
                # for key in preweight.keys():
                #     preweight[key] = (1 - 0.001) * preweight[key] + (0.001) * model.state_dict()[key]
                # premodel.load_state_dict(preweight)
                # pro1, _, _ = premodel(x1)
                # pro2, _, _ = premodel(x2)
                # log_probs= model(images)
                loss1 = self.criterion(log_probs, labels)
                # loss2 = 0.5 * ((F.cosine_similarity(pre1, pro2, dim=-1).mean() + F.cosine_similarity(pre2, pro1, dim=-1).mean()) / 2).__abs__()
                loss2 = 0
                loss = loss1+loss2
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} Loss1: {:.6f} Loss2: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(),loss1.item(),loss2))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                # batch_loss1.append(loss1.item())
                # batch_loss2.append(loss2.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            # epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            # print("epoch loss: {:.6f} loss1: {:.6f} loss2: {:.6f}".format(sum(epoch_loss) / len(epoch_loss),
            #                                                               sum(epoch_loss1) / len(epoch_loss1),
            #                                                               sum(epoch_loss2) / len(epoch_loss2)))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            pro,pre,outputs = model(images)
            # outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    if args.gpu == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # print(labels)
        # Inference
        pro, pre, outputs = model(images)
        # outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        # print(pred_labels)
        # print(labels)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
