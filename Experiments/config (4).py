class DefaultConfigs(object):
    n_classes = 192  ## number of classes
    img_weight = 224  ## image width
    img_height = 224  ## image height
    batch_size = 16 ## batch size
    epochs = 36    ## epochs
    learning_rate=0.0001  ## learning rate
config = DefaultConfigs()

''' Optimizer uses lr = 0.05
    lr = 0.05 if epoch < 30
    lr = 0.005 if 30 <= epoch < 60
    lr = 0.0005 if 60 <= epoch < 90
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

'''

#class AlexNetConfig(object):
#    n_classes = 192
#    img_weight = 224
#    img_height = 224
#    learning_rate = 0.001
#    batch_size = 32
#    epochs = 50