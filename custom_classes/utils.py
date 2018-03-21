import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from threading import Thread



def convert_to_onehot(Y, num_classes, include_0 = False):  #convert a [b] label vector to a [b,number_of_categories] tensor in one_hot form
    one_hot = np.zeros((Y.shape[0], num_classes))
    for i in range(num_classes):
        one_hot[:,i:i+1] = (np.expand_dims(Y,1).astype(np.int) == i) if include_0 else (np.expand_dims(Y,1).astype(np.int) == i+1)     
    return one_hot


def plot_learning_curve(train_loss, val_loss, val_acc, validation_interval, mean_window = 10):
    
    def running_mean(x, N):
        if len(x) < N:
            return [np.nan]
        cumsum = np.cumsum(np.insert(x,0,0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 
    
    training_running_loss = running_mean(train_loss, mean_window)
    
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(arange(len(train_loss)), train_loss, 'b')
    ax1.plot(mean_window -1 + arange(len(training_running_loss)), training_running_loss, 'c')
    ax1.plot(validation_interval * arange(len(val_loss)), (val_loss), 'r')
    ax2.plot(validation_interval * arange(len(val_acc)), val_acc, 'g')
    
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax1.set_ylim(0,3)

    ax2.set_ylim(0,1)
    ax2.set_ylabel('validation Accuracy')
    
    
    if ((val_acc != []) and (val_loss !=[])):
        best_validation_acc_idx = np.argmax(val_acc)
        best_validation_loss_idx = np.argmin(val_loss)
        best_training_running_loss_idx = np.argmin(training_running_loss)
    
        best_validation_acc = val_acc[best_validation_acc_idx]
        best_validation_loss = val_loss[best_validation_loss_idx]
        best_training_running_loss = training_running_loss[best_training_running_loss_idx]
        
        
        ax1.plot((mean_window -1 + best_training_running_loss_idx,
                  mean_window -1 + best_training_running_loss_idx), (0, 10), 'c--')
        ax1.plot((0, mean_window -1 + len(training_running_loss) ),
                 (best_training_running_loss, best_training_running_loss), 'c--')
        

        ax1.plot((validation_interval * best_validation_loss_idx,
                  validation_interval * best_validation_loss_idx), (0, 10), 'r--')
        ax1.plot((0, validation_interval * len(val_loss) ),
                 (best_validation_loss, best_validation_loss), 'r--')

        ax2.plot((validation_interval * best_validation_acc_idx,
                  validation_interval * best_validation_acc_idx), (0, 10), 'g--')
        ax2.plot((0, validation_interval * len(val_acc) ),
                 (best_validation_acc, best_validation_acc), 'g--')
     
    plt.show()
    
def start_thread(func, args=tuple()):
    t = Thread(target=func, args=args)
    t.deamon = True
    t.start()
    return t
