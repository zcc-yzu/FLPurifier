import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from .models.models import CNNFashion_Mnist, WideResNet
from .utils.utils import get_dataset, average_weights, exp_details
from data_loader import DatasetForN,dataForN

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    if args.gpu == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    train_dataset = dataForN(train_dataset,args)
    test_BA = DatasetForN(train_dataset,9,10,ifaddtrig=True)
    # test_BA.addTrigger(0,3,3)
    # test_BA.dataforTest(0)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            print("lao")
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            # global_model = CNNCifar(args=args)
            global_model = WideResNet(depth=16, num_classes=10, widen_factor=1, dropRate=0)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        for idx in range(10):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=idx, logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        # global_weights = KD(args=args,local_weights=local_weights,global_weights=global_weights,global_round=epoch,trainloader=KDloader)
        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(10):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=c, logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        torch.save(global_model.state_dict(), 'BA2net_parameters.pt')
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        # test_acc, test_loss = test_inference(args, global_model, test_dataset)
        # print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        test_ba, test_BAL= test_inference(args, global_model, test_BA)
        print("|---- BA Accuracy: {:.2f}%".format(100 * test_ba))
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

