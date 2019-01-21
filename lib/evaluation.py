import torch
import numpy as np
import glob
from lib.train import Train

def plot_train_val(patterns, fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    dummy_trainer = Train(None, None, None, None)
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['train_losses']
            dummy_trainer.all_losses = l
            plt.plot(dummy_trainer.smooth_loss(), label=name)
            
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['val_losses']
            dummy_trainer.val_losses = l
            plt.plot(dummy_trainer.smooth_valloss(), label=name)

    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
def plot_fill(lines, x=None, color='b', label='default'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for l in lines:
        if x is not None:
            plt.plot(x, l, color=color, alpha=0.2)
        else:
            plt.plot(l, color=color, alpha=0.2)
    
    # lines may not have the same length
    max_length = max([len(l) for l in lines])
    middle_line = np.zeros(max_length)    
    for i in range(max_length):
        middle_line[i] = np.percentile([l[i] for l in lines if len(l) > i], 50)
        
    if x is not None:
        plt.plot(x, middle_line, color=color, label=label)
    else:
        plt.plot(middle_line, color=color, label=label)
    
def get_train_val_curves(pattern):
    dummy_trainer = Train(None, None, None, None)
    tr_curves = []
    val_curves = []
    name = ""
    for cpt_fn in glob.glob(pattern):
        cpt = torch.load(cpt_fn)
        name = cpt_fn.split('.')[0].split('/')[-1]
        dummy_trainer.all_losses = cpt['train_losses']
        dummy_trainer.val_losses = cpt['val_losses']
        
        tr_curves.append(dummy_trainer.smooth_loss())
        val_curves.append(dummy_trainer.smooth_valloss())
    return tr_curves, val_curves, name

def plot_train_val_multiple(patterns, colors=['blue', 'orange', 'green', 'red', 
                                              'purple', 'brown', 'pink', 'gray'], 
                            fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(tr_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(val_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()

        
                                
        


