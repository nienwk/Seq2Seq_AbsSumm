import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence
from torchtext.vocab import Vocab
from torchmetrics.text.rouge import ROUGEScore

from numpy import mean as npmean

import argparse
import os
from textwrap import dedent
from typing import Tuple

from seq2seq_text_summarization.helpers.training import setup_metrics_dict, compute_metrics, collate_metrics, append_metrics, print_metrics
from seq2seq_text_summarization.helpers.save_utility import save_checkpoint
from seq2seq_text_summarization.helpers.rng import generator_obj_seed

from seq2seq_text_summarization.configs.data_prep_configs import POSTPROCESSING_DIR, TRAIN_CSV, VAL_CSV
from seq2seq_text_summarization.configs.dataloader_configs import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, BUCKET_MULTIPLIER, TOKENIZER, LANGUAGE, PAD_TOKEN, VOCAB_DIR, SOURCE_VOCAB_EXPORT, TARGET_VOCAB_EXPORT
from seq2seq_text_summarization.configs.loss_configs import LABEL_SMOOTHING
from seq2seq_text_summarization.configs.optimizer_configs import OPTIMIZER, INITIAL_LEARNING_RATE, WEIGHT_DECAY, SGD_CONFIG, ADAM_CONFIG
from seq2seq_text_summarization.configs.scheduler_configs import MILESTONE_RATIO, MILESTONE_HARD, LINEAR_LR_CONFIG, COSINE_ANNEALING_CONFIG
from seq2seq_text_summarization.configs.train_configs import NUM_EPOCHS, ROUGE_KEYS, VALID_FREQ, PRINT_FREQ, SAVE_FREQ

from seq2seq_text_summarization.dataset.AFFDataset import AmazonFineFoodDataset

from seq2seq_text_summarization.dataloaders.bucket_sampler import BucketSampler
from seq2seq_text_summarization.dataloaders.collate_batch import collate_batch
from seq2seq_text_summarization.dataloaders import vocab as V

from seq2seq_text_summarization.models.encoders import *
from seq2seq_text_summarization.models.decoders import *
from seq2seq_text_summarization.models import *

def train_epoch(
    args: argparse.ArgumentParser,
    epoch_count: int,
    net: nn.Module,
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    trainloader: DataLoader,
    validloader: DataLoader,
    vocabulary: Vocab,
    preEpoch_generator_state: torch.Tensor,
    train_loss: list[float],
    val_loss: list[float],
    train_metrics: dict[str, list[float]],
    val_metrics: dict[str, list[float]],
    scheduler=None,
    verbose: bool = True,
    val_freq: int = VALID_FREQ,
    rouge_keys: Tuple[str] = ROUGE_KEYS,
    save_freq: int = -1,
    save_slot: int = None,
    print_freq: int = PRINT_FREQ,
    prev_iter: int = None,
    ) -> Tuple[list[float], list[float], dict[str, list[float]], dict[str, list[float]]]:

    rouge=ROUGEScore(rouge_keys=rouge_keys)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print('Epoch: %d' % epoch_count)

    best_val_loss = min(val_loss) if (val_loss!=[]) else 1e9

    max_num_iters = len(trainloader)

    for batch_idx, (text, summ) in enumerate(trainloader):
        # For loading and passing the iterations, since seed state of trainloader is only incremented once per epoch for the shuffling in BatchSampler.
        if (not (prev_iter is None)) and (batch_idx+1 < prev_iter):
            continue
        net.train()

        # text, summ are packed_sequences
        text, summ = text.to(device), summ.to(device)

        optimizer.zero_grad()

        # model should produce a 2-tuple.
        # output_seq should be a list of lists, where it is ordered by the batch,
        # and each inner list contains the token indices (must be passable into vocabulary.lookup_tokens() method)
        # of an output sequence.
        # output_logits shape = (batch_size, vocab_size, longest_seq_len)
        output_seq, output_logits = net(text, summ)

        # pad the packed sequence, retrieve the Tensor
        tmp_summ, _ = pad_packed_sequence(summ, batch_first=True, padding_value=0)
        # tmp_summ = (batch_size, longest_seq_len)
        tmp_summ = tmp_summ.long() # convert to long for criterion computation
        loss = criterion(output_logits, tmp_summ)
        loss.backward()
        optimizer.step()
        if not(scheduler is None):
            scheduler.step()

        train_loss.append(loss.item())

        # Compute metrics via helper function
        train_metrics = compute_metrics(train_metrics,vocabulary,rouge,output_seq,summ)

        if verbose and (batch_idx+1) % print_freq == 0:
            print("-"*50)
            print(f"iteration : {batch_idx+1:3d}, culmulative average training loss : {npmean(train_loss):0.4f}")
            print_metrics(train_metrics)

        
        if (((batch_idx+1) % val_freq) == 0) or ((batch_idx+1)==max_num_iters):
            eval_loss, eval_metrics = eval(epoch_count, batch_idx+1, max_num_iters, net, criterion, validloader, vocabulary, verbose, rouge, rouge_keys)
            val_loss.append(eval_loss)
            val_metrics = append_metrics(eval_metrics, val_metrics)
            if verbose:
                print(f"iteration : {batch_idx+1:3d}, culmulative average validation loss : {npmean(val_loss):0.4f}")
                print_metrics(val_metrics)
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                save_checkpoint(
                    args=args,
                    model=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    curr_trainloader_seed_state=preEpoch_generator_state,
                    curr_pytorch_seed_state=torch.get_rng_state(),
                    training_loss=train_loss,
                    validation_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch=epoch_count,
                    iter=batch_idx+1,
                    save_slot=save_slot,
                    verbose=verbose,
                    isBest=True,
                    max_num_iters_epoch=max_num_iters,
                )

        if ((save_freq!=(-1)) and (((batch_idx+1) % save_freq) != 0)) or ((batch_idx+1)==max_num_iters):
            save_checkpoint(
                    args=args,
                    model=net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    curr_trainloader_seed_state=preEpoch_generator_state,
                    curr_pytorch_seed_state=torch.get_rng_state(),
                    training_loss=train_loss,
                    validation_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch=epoch_count,
                    iter=batch_idx+1,
                    save_slot=save_slot,
                    verbose=verbose,
                    isBest=False,
                    max_num_iters_epoch=max_num_iters,
                )

    # val_loss and val_metrics contain (ceiling(max_num_iters / save_freq)) validation values, where the last one may be additional at the end of the epoch 
    return train_loss, val_loss, train_metrics, val_metrics
 
def eval(
    epoch_count: int,
    iter_count: int,
    max_num_iters: int,
    net: nn.Module,
    criterion: nn.CrossEntropyLoss,
    testloader: DataLoader,
    vocabulary: Vocab,
    verbose: bool,
    rouge: ROUGEScore,
    rouge_keys: Tuple,
    )-> Tuple[float, dict]:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.eval()
    test_loss = []
    metrics = setup_metrics_dict(rouge_keys)

    with torch.inference_mode():
        if verbose:
            print("-"*50)
            print(f'Validating iteration {iter_count:6d}/{max_num_iters:6d} of epoch {epoch_count+1:4d}...')
        for _, (text, summ) in enumerate(testloader):
            # text, summ are packed_sequences
            text, summ = text.to(device), summ.to(device)

            # model should produce a 2-tuple.
            # output_seq should be a list of lists, where it is ordered by the batch,
            # and each inner list contains the token indices (must be passable into vocabulary.lookup_tokens() method)
            # of an output sequence.
            # output_logits shape = (batch_size, vocab_size, longest_seq_len)
            output_seq, output_logits = net(text, summ)

            # pad the packed sequence, retrieve the Tensor
            tmp_summ, _ = pad_packed_sequence(summ, batch_first=True, padding_value=0)
            # tmp_summ = (batch_size, longest_seq_len)
            tmp_summ = tmp_summ.long() # convert to long for criterion computation
            loss = criterion(output_logits, tmp_summ)
            test_loss.append(loss.item())

            # Compute metrics via helper function
            metrics = compute_metrics(metrics,vocabulary,rouge,output_seq,summ)

    metrics = collate_metrics(metrics)

    return npmean(test_loss), metrics


def main(args):
    if type(args.load) == str:
        assert args.load in os.listdir(f'./saves/'), f"save {args.load} is not found!"

        checkpoint = torch.load(f'./saves/{args.load}.pt')

        # Load previous input arguments, overrides all current arguments except verbosity
        new_verbose = args.verbose
        args = checkpoint['args']
        args.verbose = new_verbose
        del new_verbose

        # Do not overwrite previous save file for starting training again
        while True:
            if (f'save{args.save}_latest.pt' in os.listdir(f'./saves/')) \
                or (f'save{args.save}_best.pt' in os.listdir(f'./saves/')):
                args.save += 1
                continue
            else:
                print(f"Save file will be saved in save slot {args.save} instead!")
                print("Remember to move the save file back to the correct slot to enable proper loading next time.")
                break

        cudnn.benchmark = args.benchmark
        cudnn.deterministic = args.deterministic
        # torch.use_deterministic_algorithms(args.deterministic)

        # Set the pytorch seed
        if not (args.pytorch_seed is None):
            _ = torch.set_rng_state(checkpoint['pytorch_seed'])

        # Load tokenizer and vocabulary
        tokenizer = V.setup_tokenizer(args.tokenizer, args.language)
        source_vocabulary = V.import_source_vocab(path=args.source_vocab_path, verbose=args.verbose)
        if args.share_vocab:
            vocabulary = source_vocabulary
        else:
            raise NotImplementedError("Separated vocabulary is not supported.")

        # Import datasets
        train_dataset = AmazonFineFoodDataset(tokenizer, vocabulary, args.train_data_path) # assumes the names of columns are correct. Possible to break here!
        valid_dataset = AmazonFineFoodDataset(tokenizer, vocabulary, args.valid_data_path) # assumes the names of columns are correct. Possible to break here!

        # Set the trainloader generator's seed
        trainloader_generator = torch.Generator().set_state(checkpoint['trainloader_seed'])
        
        # Load training, validation datasets into dataloaders
        trainloader = DataLoader(train_dataset,batch_sampler=BucketSampler(train_dataset, args.train_batchsize, trainloader_generator, bucket_multiplier=args.bucket_multiplier, drop_last=True, shuffle=True), collate_fn=collate_batch, num_workers=args.num_workers)
        validloader = DataLoader(valid_dataset,batch_sampler=BucketSampler(valid_dataset, args.val_batchsize, drop_last=False, shuffle=False), collate_fn=collate_batch, num_workers=args.num_workers)

        max_iter_len = len(trainloader)*args.epochs
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO Load previous model state, i.e. parameters
        if args.model == "model1":
            net = Model1(Encoder1(), Decoder1()).to(device)
            net.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise NotImplementedError(f"model is not implemented! Got model name {args.model}")

        # Use CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary([args.pad_token])[0], label_smoothing=args.smoothing).to(device)

        # Setup optimizer
        if args.optimizer == 'SGD':
            config = {
                'lr': args.learning_rate,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'dampening': args.dampening,
            }
            # Load previous SGD state, i.e. parameters
            optimizer = SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], dampening=config['dampening'], weight_decay=config['weight_decay'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif args.optimizer == 'Adam':
            config = {
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
                'b1': args.beta1,
                'b2': args.beta2,
            }
            # Load previous Adam state, i.e. parameters
            optimizer = Adam(net.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']), weight_decay=config['weight_decay'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            raise NotImplementedError(f"Optimizer choice is not supported: {args.optimizer}")

        # Load scheduler state
        if args.scheduler:
            milestone = min(int((args.milestone_ratio * max_iter_len)// 1.0), args.milestone_hard)
            scheduler = SequentialLR(optimizer,[LinearLR(optimizer, args.warmup_start_factor, total_iters=milestone), CosineAnnealingLR(optimizer, max_iter_len-milestone, args.min_lr)], [milestone])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler = None
        
        prev_epoch = checkpoint['epoch'] # starts from 1
        prev_iter = checkpoint['iter'] # starts from 1
        training_loss = checkpoint['histories']["train_loss"]
        validation_loss = checkpoint['histories']["valid_loss"]
        training_metrics = checkpoint['train_metrics']
        validation_metrics = checkpoint['val_metrics']

    else:
        if not (args.pytorch_seed is None):
            _ = torch.manual_seed(args.pytorch_seed)

        cudnn.benchmark = args.benchmark
        cudnn.deterministic = args.deterministic
        # torch.use_deterministic_algorithms(args.deterministic)

        # Generator for trainloader, use to set generator state in resuming training
        trainloader_generator = generator_obj_seed(args.trainloader_seed)

        # Load tokenizer and vocabulary
        tokenizer = V.setup_tokenizer(args.tokenizer, args.language)
        source_vocabulary = V.import_source_vocab(path=args.source_vocab_path, verbose=args.verbose)
        if args.share_vocab:
            vocabulary = source_vocabulary
        else:
            raise NotImplementedError("Separated vocabulary is not supported.")

        # Import datasets
        train_dataset = AmazonFineFoodDataset(tokenizer, vocabulary, args.train_data_path) # assumes the names of columns are correct. Possible to break here!
        valid_dataset = AmazonFineFoodDataset(tokenizer, vocabulary, args.valid_data_path) # assumes the names of columns are correct. Possible to break here!

        # Load training, validation datasets into dataloaders
        trainloader = DataLoader(train_dataset,batch_sampler=BucketSampler(train_dataset, args.train_batchsize, trainloader_generator, bucket_multiplier=args.bucket_multiplier, drop_last=True, shuffle=True), collate_fn=collate_batch, num_workers=args.num_workers)
        validloader = DataLoader(valid_dataset,batch_sampler=BucketSampler(valid_dataset, args.val_batchsize, drop_last=False, shuffle=False), collate_fn=collate_batch, num_workers=args.num_workers)

        max_iter_len = len(trainloader)*args.epochs

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO Setup model
        if args.model == "model1":
            net = Model1(Encoder1(), Decoder1()).to(device)
        else:
            raise NotImplementedError(f"model is not implemented! Got model name {args.model}")

        # Use CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary([args.pad_token])[0], label_smoothing=args.smoothing).to(device)

        # Setup optimizer
        if args.optimizer == 'SGD':
            config = {
                'lr': args.learning_rate,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'dampening': args.dampening,
            }
            optimizer = SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'], dampening=config['dampening'], weight_decay=config['weight_decay'])
        elif args.optimizer == 'Adam':
            config = {
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
                'b1': args.beta1,
                'b2': args.beta2,
            }
            optimizer = Adam(net.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']), weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError(f"Optimizer choice is not supported: {args.optimizer}")

        # Setup scheduler
        if args.scheduler:
            milestone = min(int((args.milestone_ratio * max_iter_len)// 1.0), args.milestone_hard)
            scheduler = SequentialLR(optimizer,[LinearLR(optimizer, args.warmup_start_factor, total_iters=milestone), CosineAnnealingLR(optimizer, max_iter_len-milestone, args.min_lr)], [milestone])
        else:
            scheduler = None

        training_loss = []
        validation_loss = []
        training_metrics = setup_metrics_dict(tuple(args.rouge_keys.strip().split()))
        validation_metrics = setup_metrics_dict(tuple(args.rouge_keys.strip().split()))


    # Set starting epoch
    try:
        type(prev_epoch)
    except NameError:
        start = 1
        prev_iter = None
    else:
        start = prev_epoch

    for epoch in range(start, args.epochs+1):
        seed_state = trainloader_generator.get_state()
        training_loss, validation_loss, training_metrics, validation_metrics = train_epoch(args, epoch, net, criterion, optimizer, trainloader, validloader, vocabulary, seed_state, training_loss, validation_loss, training_metrics, validation_metrics, scheduler, args.verbose, args.valid_freq, tuple(args.rouge_keys.strip().split()), args.save_freq, args.save, args.print_freq, prev_iter)

        if args.verbose:
            print("-"*50)
            print(f"Epoch {epoch:4d}/{args.epochs}, culmulative training loss : {npmean(training_loss):0.4f}, culmulative validation loss : {npmean(validation_loss):0.4f}")
            print("-"*50)
            print("TRAINING METRICS")
            print_metrics(training_metrics)
            print("-"*50)
            print("VALIDATION METRICS")
            print_metrics(validation_metrics)
            print("-"*50)

def debug_parser(args):
    print(args)

if __name__ == "__main__":
    """ Launch with -vdu --no-benchmark for no benchmark, deterministic, verbose and with scheduler
    """
    # Parser to allow modifying attributes from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent(
    '''Run the Seq2Seq model.
    '''), allow_abbrev=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='show progress as model trains (default: False)', default=False)
    # Reproducibility
    parser.add_argument('--pytorch-seed', metavar='<SEED>', type=int, help='set pytorch initial seed (default: None)', default=None)
    parser.add_argument('--trainloader-seed', metavar='<SEED>', type=int, help='set training dataset Dataloader initial seed (default: None)', default=None)
    parser.add_argument('-b', '--benchmark', action=argparse.BooleanOptionalAction, help='set torch.backends.cudnn.benchmark=True for optimization. Set to False for reproducibility at the expense of code optimization.', default=True)
    parser.add_argument('-d', '--deterministic', action=argparse.BooleanOptionalAction, help='use deterministic algorithms', default=False)
    # Dataset
    parser.add_argument('-sv', '--source-vocab-path', metavar='<PATH>', type=str, help=f'path to source vocab (default: {VOCAB_DIR+SOURCE_VOCAB_EXPORT})', default=VOCAB_DIR+SOURCE_VOCAB_EXPORT)
    parser.add_argument('-tv', '--target-vocab-path', metavar='<PATH>', type=str, help=f'path to target vocab (default: {VOCAB_DIR+TARGET_VOCAB_EXPORT})', default=VOCAB_DIR+TARGET_VOCAB_EXPORT)
    parser.add_argument('--train-data-path', metavar='<PATH>', type=str, help=f'path to training dataset (default: {POSTPROCESSING_DIR+TRAIN_CSV})', default=POSTPROCESSING_DIR+TRAIN_CSV)
    parser.add_argument('--valid-data-path', metavar='<PATH>', type=str, help=f'path to validation dataset (default: {POSTPROCESSING_DIR+VAL_CSV})', default=POSTPROCESSING_DIR+VAL_CSV)
    parser.add_argument('--share-vocab', action=argparse.BooleanOptionalAction, help=f'choice to share vocabulary between source and target', default=True)
    parser.add_argument('-t', '--tokenizer', metavar='<NAME>', type=str, help=f'choice of tokenizer (default: {TOKENIZER})', default=TOKENIZER)
    parser.add_argument('-l', '--language', metavar='<NAME>', type=str, help=f'choice of language (default: {LANGUAGE})', default=LANGUAGE)
    parser.add_argument('--pad-token', metavar='<NAME>', type=str, help=f'choice of pad token (default: {PAD_TOKEN})', default=PAD_TOKEN)
    # Dataloader
    parser.add_argument('--num-workers', metavar='<COUNT>', type=int, help=f'set the number of processes for data loading multiprocessing (default: {NUM_WORKERS})', default=NUM_WORKERS)
    parser.add_argument('-e', '--epochs', metavar='<COUNT>', type=int, help=f'set the number of episodes to run (default: {NUM_EPOCHS})', default=NUM_EPOCHS)
    parser.add_argument('--train-batchsize', metavar='<COUNT>', type=int, help=f'set the batch size of the training dataset (default: {TRAIN_BATCH_SIZE})', default=TRAIN_BATCH_SIZE)
    parser.add_argument('--val-batchsize', metavar='<COUNT>', type=int, help=f'set the batch size of the validation dataset (default: {VAL_BATCH_SIZE})', default=VAL_BATCH_SIZE)
    parser.add_argument('--bucket-multiplier', metavar='<COUNT>', type=int, help=f'set the bucket multiplier for BucketSampler (default: {BUCKET_MULTIPLIER})', default=BUCKET_MULTIPLIER)
    # Optimizer
    parser.add_argument('-o', '--optimizer', metavar='<NAME>', type=str, help=f'choice of optimizer (default: {OPTIMIZER})', default=OPTIMIZER)
    parser.add_argument('-lr', '--learning-rate', metavar='<RATE>', type=float, help=f'set the learning rate (default: {INITIAL_LEARNING_RATE})', default=INITIAL_LEARNING_RATE)
    parser.add_argument('-w', '--weight-decay', metavar='<DECAY>', type=float, help=f'set the weight decay (default: {WEIGHT_DECAY})', default=WEIGHT_DECAY)
    # Optimizer: SGD specific
    parser.add_argument('-m', '--momentum', metavar='<MOMENTUM>', type=float, help=f'set SGD momentum (default: {SGD_CONFIG["momentum"]})', default=SGD_CONFIG['momentum'])
    parser.add_argument('-s', '--dampening', metavar='<DAMPENING>', type=float, help=f'set SGD dampening (default: {SGD_CONFIG["dampening"]})', default=SGD_CONFIG["dampening"])
    # Optimizer: Adam specific
    parser.add_argument('-b1', '--beta1', metavar='<BETA_1>', type=float, help=f'set the beta 1 of Adam (default: {ADAM_CONFIG["betas"][0]})', default=ADAM_CONFIG["betas"][0])
    parser.add_argument('-b2', '--beta2', metavar='<BETA_2>', type=float, help=f'set the beta 2 of Adam (default: {ADAM_CONFIG["betas"][1]})', default=ADAM_CONFIG["betas"][1])
    # Scheduler
    parser.add_argument('-u', '--scheduler', action='store_true', help='use cosine annealing scheduler with warmup (default: False)', default=False)
    parser.add_argument('--milestone-ratio', metavar='<RATIO>', type=float, help=f'set the milestone ratio to control transition between warmup and cosine annealing learning rate schedule (default: {MILESTONE_RATIO}). Will take the smaller between (milestone ratio)*max_iter and milestone_hard', default=MILESTONE_RATIO)
    parser.add_argument('--milestone-hard', metavar='<COUNT>', type=int, help=f'set the milestone hard limit to control transition between warmup and cosine annealing learning rate schedule (default: {MILESTONE_HARD}). Will take the smaller between (milestone ratio)*max_iter and milestone_hard', default=MILESTONE_HARD)
    parser.add_argument('--warmup-start-factor', metavar='<MULTIPLIER>', type=float, help=f'set the learning rate warmup starting learning rate to be (initial learning rate * <MULTIPLIER>) (default: {LINEAR_LR_CONFIG["start_factor"]})', default=LINEAR_LR_CONFIG["start_factor"])
    parser.add_argument('--min-lr', metavar='<RATE>', type=float, help=f'set the minimum learning rate to decay to in cosine annealing learning rate schedule (default: {COSINE_ANNEALING_CONFIG["eta_min"]})', default=COSINE_ANNEALING_CONFIG["eta_min"])
    # Loss and metrics
    parser.add_argument('--smoothing', metavar='<SMOOTHING>', type=float, help=f'set the label smoothing value for CrossEntropyLoss (default: {LABEL_SMOOTHING})', default=LABEL_SMOOTHING)
    parser.add_argument('--rouge-keys', metavar='<ROUGE-KEYS>', type=str, help=f'set the ROUGE keys to use to compute metrics (default: {" ".join(ROUGE_KEYS)})', default=" ".join(ROUGE_KEYS))
    parser.add_argument('-vf','--valid-freq', metavar='<COUNT>', type=int, help=f'set the validation frequency, in iterations. (default: {VALID_FREQ})', default=VALID_FREQ)
    parser.add_argument('-pf','--print-freq', metavar='<COUNT>', type=int, help=f'set the printing frequency, in iterations. (default: {PRINT_FREQ})', default=PRINT_FREQ)
    # Model
    # TODO change to a default baseline model
    parser.add_argument('-n', '--model', metavar='<NAME>', type=str, help='use model with name. (default: None)', default=None)
    # Save & Loading
    parser.add_argument('--save', metavar='<SLOT>', type=int, help='save checkpoint to save slot <SLOT> in ./saves/ (default: None)', default=None)
    parser.add_argument('--save-freq', metavar='<COUNT>', type=int, help=f'set number of iterations before saving to save slot <SLOT> in ./saves/ (default: {SAVE_FREQ})', default=SAVE_FREQ)
    parser.add_argument('--load', metavar='<NAME>', type=str, help='load checkpoint <NAME> from ./saves/ (default: None)', default=None)
    args = parser.parse_args()

    debug_parser(args)
    #main(args)