from argparse import ArgumentParser
from typing import Union
from torch.nn import Module
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor, save
from os import mkdir, rename
from os.path import isdir
from .training import collate_metrics_all

def save_checkpoint(
    args : ArgumentParser,
    model : Module,
    optimizer : Union[SGD, Adam],
    scheduler : Union[CosineAnnealingLR, None],
    curr_trainloader_seed_state : Tensor,
    curr_pytorch_seed_state: Tensor,
    training_loss : list,
    validation_loss : list,
    train_metrics: dict,
    val_metrics: dict,
    epoch : int,
    iter : int,
    save_slot : int,
    verbose : bool,
    isBest: bool=False,
    max_num_iters_epoch: int=None,
    ):
    """ Checkpoint saving utility.\n
    Bundles model `state_dict`, optimizer `state_dict`, trainloader seed state, PyTorch seed state, training/validation histories, epoch/iter count of checkpoint, and lastly scheduler `state_dict` if applicable.\n
    Set the `isBest` flag to `True` to only save as the best checkpoint. Otherwise saves as 'latest' and 'epoch{`epoch`}_iter{`iter`}' checkpoints. Default: `isBest`=`False`
    """
    assert type(save_slot)==int, f"Save slot must be integer! Got {type(save_slot)}"
    if verbose:
        print("-"*50)
        print(f'Saving checkpoint at iteration {iter}/{max_num_iters_epoch} of epoch {epoch} in save slot {save_slot}...')
    state = {
        'args': args,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'trainloader_seed': curr_trainloader_seed_state,
        'pytorch_seed': curr_pytorch_seed_state,
        'histories': {
            "train_loss": training_loss,
            "valid_loss": validation_loss,
        },
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'epoch': epoch,
        'iter': iter,
    }
    if scheduler != None:
        state['scheduler_state_dict'] = scheduler.state_dict()

    if not isdir('saves'):
        mkdir('saves')
    
    if isBest:
        save(state, f'./saves/save{save_slot}_best.pt')
    else:
        # save(state, f'./saves/save{save_slot}_epoch{epoch}_iter{iter}.pt')
        try:
            rename(f'./saves/save{save_slot}_latest.pt', f'./saves/save{save_slot}_prevLatest.pt')
        except FileNotFoundError:
            pass
        finally:
            save(state, f'./saves/save{save_slot}_latest.pt')

def save_epoch_loss_and_metrics():
    pass

def save_test(test_loss : float, metrics : dict, save_slot : Union[int, None] = None, verbose : bool = False):
    """Simple test result saving utility for easier access via `torch.load`.
    """
    if not isdir('test_results'):
        mkdir('test_results')
    
    tmp_metrics = collate_metrics_all(metrics)

    if save_slot != None:
        save({"metrics" : tmp_metrics, "test_loss" : test_loss}, f"./test_results/result{save_slot}.pt")

    if verbose:
        print(f"\nTesting loss : {test_loss}.")
        for k,v in tmp_metrics:
            print(f"{k} : {v}.")