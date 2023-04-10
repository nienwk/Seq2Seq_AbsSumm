import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore
from numpy import mean as npmean
from seq2seq_text_summarization.configs.data_prep_configs import POSTPROCESSING_DIR, TEST_CSV
from seq2seq_text_summarization.configs.dataloader_configs import NUM_WORKERS, SOURCE_VOCAB_EXPORT, TARGET_VOCAB_EXPORT, TEST_BATCH_SIZE, VOCAB_DIR
from seq2seq_text_summarization.configs.model_configs import MODEL1_NAME
from seq2seq_text_summarization.dataloaders.bucket_sampler import BucketSampler
from seq2seq_text_summarization.dataloaders.collate_batch import collate_batch
from seq2seq_text_summarization.dataloaders.vocab import Vocab
from seq2seq_text_summarization.dataloaders import vocab as V
from seq2seq_text_summarization.dataset.AFFDataset import AmazonFineFoodDataset
from seq2seq_text_summarization.helpers.save_utility import save_test
from seq2seq_text_summarization.helpers.training import setup_metrics_dict, compute_metrics, collate_metrics_all

import argparse
from textwrap import dedent
from typing import Tuple

from seq2seq_text_summarization.models.model1 import Model1

def eval(
    net: nn.Module,
    criterion: nn.CrossEntropyLoss,
    testloader: DataLoader,
    vocabulary: Vocab,
    vocab_size: int,
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
            print(f'Validating test dataset ...')
            print("-"*50)
        for _, (text, summ) in enumerate(testloader):
            # text, summ are packed_sequences
            text, summ = text.to(device), summ.to(device)

            # model should produce a 2-tuple.
            # output_seq should be a list of lists, where it is ordered by the batch,
            # and each inner list contains the token indices (must be passable into vocabulary.lookup_tokens() method)
            # of an output sequence.
            # output_logits should be a list of batch_size Tensors of shape = (seq_len, vocab_size)
            # We ignore attn_weights in training and eval
            output_seq, output_logits, _ = net(text)

            # pad the packed sequence, retrieve the Tensor
            padded_target_summs, target_lengths = pad_packed_sequence(summ, batch_first=True, padding_value=0)
            # padded_target_summs = (batch_size, longest_seq_len)
            padded_target_summs = padded_target_summs.long() # convert to long for criterion computation

            batch_max_target_len = max(target_lengths).item()
            padded_output_logits = []
            
            for logit in output_logits:
                if logit.size(0) < batch_max_target_len:
                    len_diff = batch_max_target_len - logit.size(0)
                    logit_padding = torch.zeros(len_diff, vocab_size).requires_grad_(False).to(device)
                    padded_output_logits.append(torch.vstack((logit, logit_padding)).contiguous())
                else:
                    padded_output_logits.append(logit[:batch_max_target_len])
            padded_output_logits = torch.stack(padded_output_logits, dim=0).permute(0,2,1)
            
            loss = criterion(padded_output_logits, padded_target_summs)
            test_loss.append(loss.item())

            # Compute metrics via helper function
            metrics = compute_metrics(metrics,vocabulary,rouge,output_seq,summ)

    metrics = collate_metrics_all(metrics)

    return npmean(test_loss), metrics

def main(args):
    if type(args.load) == str:
        assert args.load in os.listdir(f'./saves/'), f"save {args.load} is not found!"
    else:
        raise ValueError(f"Save filename must be a valid name.")

    checkpoint = torch.load(f'./saves/{args.load}')
    args2=checkpoint['args']
    cudnn.benchmark = args2.benchmark
    cudnn.deterministic = args2.deterministic
    # torch.use_deterministic_algorithms(args2.deterministic)

    # Set the pytorch seed
    if not (args2.pytorch_seed is None):
        _ = torch.set_rng_state(checkpoint['pytorch_seed'])

    # Load tokenizer and vocabulary
    tokenizer = V.setup_tokenizer(args2.tokenizer, args2.language)
    source_vocabulary = V.import_source_vocab(path=args.source_vocab_path, verbose=args.verbose)
    if args.share_vocab:
        vocabulary = source_vocabulary
    else:
        raise NotImplementedError("Separated vocabulary is not supported.")

    # Import datasets
    test_dataset = AmazonFineFoodDataset(tokenizer, vocabulary, vocabulary([args2.end_token]), args.test_data_path) # assumes the names of columns are correct. Possible to break here!
    
    # Load training, validation datasets into dataloaders
    testloader = DataLoader(test_dataset,batch_sampler=BucketSampler(test_dataset, args.test_batchsize, drop_last=False, shuffle=False), collate_fn=collate_batch, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load previous model state, i.e. parameters
    if args.model == MODEL1_NAME:
        net = Model1(
            vocab_size=len(vocabulary),
            padding_index=vocabulary([args2.pad_token])[0],
            start_of_summ_index=vocabulary([args2.start_token])[0],
            end_of_summ_index=vocabulary([args2.end_token])[0],
            activation=args2.model1_activation,
            embedding_dim=args2.model1_embedding_dim,
            embedding_dropout_p=args2.model1_embedding_dropout_p,
            encoder_hidden_dim=args2.model1_encoder_hidden_dim,
            encoder_num_layers=args2.model1_encoder_num_layers,
            encoder_rnn_dropout_p=args2.model1_encoder_rnn_dropout_p,
            encoder_fc_dropout_p=args2.model1_encoder_fc_dropout_p,
            bidirectional_encoder=args2.model1_encoder_bidirectional,
            decoder_hidden_dim=args2.model1_decoder_hidden_dim,
            decoder_num_layers=args2.model1_decoder_num_layers,
            decoder_rnn_dropout_p=args2.model1_decoder_rnn_dropout_p,
            decoder_num_attention_head=args2.model1_decoder_num_attention_head,
            decoder_attention_dropout_p=args2.model1_decoder_attention_dropout_p,
            decoder_input_feeding_fc_dropout_p=args2.model1_decoder_input_feeding_fc_dropout_p,
            decoder_attentional_fc_out_dim=args2.model1_decoder_attentional_fc_out_dim,
            beam_search_k=args2.model1_beam_search_k,
            generation_limit=args2.model1_generation_limit,
            hypothesis_limit=args2.model1_max_hypothesis_count,
            teacher_forcing_ratio=args2.model1_teacher_forcing_rate
        ).to(device)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise NotImplementedError(f"model is not implemented! Got model name {args.model}")

    # Use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary([args2.pad_token])[0], label_smoothing=args2.smoothing).to(device)

    rouge=ROUGEScore(rouge_keys=tuple(args2.rouge_keys.strip().split()))

    mean_loss, metrics = eval(net, criterion, testloader, vocabulary, len(vocabulary), args.verbose, rouge, tuple(args2.rouge_keys.strip().split()))

    save_test(mean_loss, metrics, args2.save, args.verbose)
    

if __name__ == "__main__":
    # Parser to allow modifying attributes from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent(
    '''Test the Seq2Seq model.
    '''), allow_abbrev=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='show progress as model trains (default: False)', default=False)
    # Dataset
    parser.add_argument('-sv', '--source-vocab-path', metavar='<PATH>', type=str, help=f'path to source vocab (default: {VOCAB_DIR+SOURCE_VOCAB_EXPORT})', default=VOCAB_DIR+SOURCE_VOCAB_EXPORT)
    parser.add_argument('-tv', '--target-vocab-path', metavar='<PATH>', type=str, help=f'path to target vocab (default: {VOCAB_DIR+TARGET_VOCAB_EXPORT})', default=VOCAB_DIR+TARGET_VOCAB_EXPORT)
    parser.add_argument('--test-data-path', metavar='<PATH>', type=str, help=f'path to testing dataset (default: {POSTPROCESSING_DIR+TEST_CSV})', default=POSTPROCESSING_DIR+TEST_CSV)
    parser.add_argument('--share-vocab', action=argparse.BooleanOptionalAction, help=f'choice to share vocabulary between source and target', default=True)
    # Dataloader
    parser.add_argument('--num-workers', metavar='<COUNT>', type=int, help=f'set the number of processes for data loading multiprocessing (default: {NUM_WORKERS})', default=NUM_WORKERS)
    parser.add_argument('--test-batchsize', metavar='<COUNT>', type=int, help=f'set the batch size of the test dataset (default: {TEST_BATCH_SIZE})', default=TEST_BATCH_SIZE)
    # Model
    parser.add_argument('-n', '--model', metavar='<NAME>', type=str, help=f'use model with name. (default: {MODEL1_NAME})', default=MODEL1_NAME)
    # Loading
    parser.add_argument('--load', metavar='<NAME>', type=str, help='load checkpoint <NAME> from ./saves/ (default: None)', default=None)
    args = parser.parse_args()

    # debug_parser(args)
    main(args)