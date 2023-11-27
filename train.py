## import libraries for training
import os
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset # Custom module for dataset handling
import timm                   # "PyTorch Image Models" for pretrained models, ResNet, EfficientNet. A collection of SOTA image models including CNNs versions, Vision Tranformers.
import matplotlib.pyplot as plt
from utils import *           # Utility functions
import warnings

# import CNNs models
from alexNetModel import ModifiedAlexNet
from torch.utils.tensorboard import SummaryWriter

# Configurations
from config import config


warnings.filterwarnings('ignore')
## Writing the loss and results
if not os.path.exists("./logs/"):
    os.mkdir("./logs/")
log = Logger()                  # Logger Setup
log.open("logs/s_log_train_2.txt")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                              |----- Train ------|----- Valid----||----- Valid----|-----------|\n')
log.write('mode      iter       epoch    |       loss       |       loss    |        mAP     |     time  |\n')
log.write('----------------------------------------------------------------------------------------------\n')

## Training the model
def train(train_loader, model, criterion, optimizer, epoch, valid_accuracy, start):
    losses = AverageMeter()
    model.train()
    model.training = True

    for i, (images, target, fnames) in enumerate(train_loader):
        img = images.cuda(non_blocking=True)
        label = target.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            logits = model(img)
        loss = criterion(logits, label)
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()        
        optimizer.zero_grad()
        scheduler.step()

        print('\r', end ='', flush=True)
        message = '%s %5.1f %6.1f        |       %0.3f     |       %0.3f     |      %0.3f     | %s' % (\
                "train", i, epoch, losses.avg, valid_accuracy[1], valid_accuracy[0], time_to_str((timer() - start),'min')) #epoch+1
        print(message , end='',flush=True)
    log.write("\n")
    log.write(message)

    return [losses.avg]


# Validating the model
def evaluate(val_loader, model, criterion, epoch, train_loss, start):
    model.cuda()
    model.eval()
    model.training = False
    map = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True) #change .to(device)
            label = target.cuda(non_blocking=True) #change .to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1) # Softmax applied on tensor along dimension 1.
            
            loss = criterion(logits, label)       # Now, losses are updated.
            losses.update(loss.item(), images.size(0))
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5, img.size(0))

            print('\r', end = '', flush=True)
            message = '%s   %5.1f %6.1f       |      %0.3f     |      %0.3f    | %s' % (\
                    "val", i, epoch, train_loss[0], losses.avg, map.avg, time_to_str((timer() - start), 'min')) #epoch+1
            print(message, end='',flush=True)
        log.write("\n")  
        log.write(message)
    return [map.avg, losses.avg]


## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5



######################## load file and get splits #############################

# Load training and validation data accordingly
train_imlist = pd.read_csv("/content/drive/My Drive/Knives/train.csv")
# Update image paths in the DataFrames
train_imlist['Id'] = train_imlist['Id'].apply(lambda x: '/content/drive/My Drive/Knives/' + x)

# Create datasets and dataloaders
train_gen = knifeDataset(train_imlist, mode="train")
train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)

# For Validation using val.csv instead of test.csv (as it was by default)
val_imlist = pd.read_csv("/content/drive/My Drive/Knives/val.csv") # CHANGED to appropriate csv file for validation, (Not validate with test.csv)
val_imlist['Id'] = val_imlist['Id'].apply(lambda x: '/content/drive/My Drive/Knives/' + x)  # -> Doc in REPORT)

val_gen = knifeDataset(val_imlist, mode="val")
val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=8)


## Loading the model to run/setup
model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=config.n_classes)

# model = create new pre-trained instance

# model = ModifiedAlexNet()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


############################# Parameters #################################


# Optimizer and scheduler setup
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.epochs * len(train_loader), eta_min=0,last_epoch=-1)
criterion = nn.CrossEntropyLoss().cuda()


# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



########################### Checkpoint ##################################

# Check for an existing checkpoint and load if found
checkpoint_path = "/content/drive/My Drive/Knives/Model_Checkpoints/last_checkpoint.pth"
#if os.path.exists(checkpoint_path):
#    checkpoint = torch.load(checkpoint_path)
#    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#    start_epoch = checkpoint['epoch']
#else:
start_epoch = 0

############################# Training #################################
scaler = torch.cuda.amp.GradScaler()
start = timer()
best_val_metric = float('inf')  # or -float('inf') for accuracy or other metrics

val_metrics = [0]

for epoch in range(0, config.epochs):
    lr = get_learning_rate(optimizer)
    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
    val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)

    # Saving the model checkpoint after each epoch
    #checkpoint = {
    #    'epoch': epoch + 1,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'scaler_state_dict': scaler.state_dict(),  # If using mixed precision
    #}
    #torch.save(checkpoint, checkpoint_path)

    ## Saving the current model
    filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
    torch.save(model.state_dict(), filename)

    ## Saving the AlexNet model
    #filename = "Knife_AlexNet_Epoch_" + str(epoch + 1) + ".pt"
    #torch.save(model.state_dict(), filename)





    # Optionally, save the best model based on validation metric
    if val_metrics[0] < best_val_metric:  # Adjust this condition based on your metric
        print(f"\nNew best metric achieved: {val_metrics[epoch]}")
        best_val_metric = val_metrics[epoch]
        best_model_path = "/content/drive/My Drive/Knives/Model_Checkpoints/best_model.pth"
        torch.save(model.state_dict(), best_model_path)
    
     # ... [any other code needed at the end of each epoch] ...


   






# ... [any code after completing all epochs, like closing loggers] ...




############################# Training #################################
#start_epoch = 0
#val_metrics = [0]
#scaler = torch.cuda.amp.GradScaler()
#start = timer()
#train
#for epoch in range(0, config.epochs):
#    lr = get_learning_rate(optimizer)
#    train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, start)
#    val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start)
    
    ## Saving the model
#    filename = "Knife-Effb0-E" + str(epoch + 1)+  ".pt"
#    torch.save(model.state_dict(), filename)
    

   
