import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from ..configs.model_configs import MODEL1_ACTIVATION, MODEL1_EMBEDDING_DIM, MODEL1_EMBEDDING_DROPOUT_P, MODEL1_ENCODER_HIDDEN_DIM, MODEL1_ENCODER_NUM_LAYERS, MODEL1_ENCODER_RNN_DROPOUT_P, MODEL1_ENCODER_FC_DROPOUT_P, MODEL1_ENCODER_BIDIRECTIONAL, MODEL1_DECODER_HIDDEN_DIM, MODEL1_DECODER_NUM_LAYERS, MODEL1_DECODER_RNN_DROPOUT_P, MODEL1_DECODER_NUM_ATTENTION_HEAD, MODEL1_DECODER_ATTENTION_DROPOUT_P, MODEL1_DECODER_INPUT_FEEDING_FC_DROPOUT_P, MODEL1_DECODER_ATTENTIONAL_FC_OUT_DIM, MODEL1_BEAM_SEARCH_K, MODEL1_GENERATION_LIMIT, MODEL1_MAX_HYPOTHESIS_COUNT, MODEL1_TEACHER_FORCING_RATE
from ..helpers.model import embedding_apply, packed_seq_apply, compute_attention_key_padding_mask, GeneratedWord
from .encoders.encoder1 import Encoder1
from .decoders.decoder1 import Decoder1

class Model1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        padding_index: int,
        start_of_summ_index: int,
        end_of_summ_index: int,

        activation: str = MODEL1_ACTIVATION,
        embedding_dim: int = MODEL1_EMBEDDING_DIM,
        embedding_dropout_p: float = MODEL1_EMBEDDING_DROPOUT_P,

        encoder_hidden_dim: int = MODEL1_ENCODER_HIDDEN_DIM,
        encoder_num_layers: int = MODEL1_ENCODER_NUM_LAYERS,
        encoder_rnn_dropout_p: float = MODEL1_ENCODER_RNN_DROPOUT_P,
        encoder_fc_dropout_p: float = MODEL1_ENCODER_FC_DROPOUT_P,
        bidirectional_encoder: bool = MODEL1_ENCODER_BIDIRECTIONAL,
        
        decoder_hidden_dim: int = MODEL1_DECODER_HIDDEN_DIM,
        decoder_num_layers: int = MODEL1_DECODER_NUM_LAYERS,
        decoder_rnn_dropout_p: float = MODEL1_DECODER_RNN_DROPOUT_P,
        decoder_num_attention_head: int = MODEL1_DECODER_NUM_ATTENTION_HEAD,
        decoder_attention_dropout_p: float = MODEL1_DECODER_ATTENTION_DROPOUT_P,
        decoder_input_feeding_fc_dropout_p: float = MODEL1_DECODER_INPUT_FEEDING_FC_DROPOUT_P,
        decoder_attentional_fc_out_dim : int = MODEL1_DECODER_ATTENTIONAL_FC_OUT_DIM,

        beam_search_k: int = MODEL1_BEAM_SEARCH_K,
        generation_limit: int = MODEL1_GENERATION_LIMIT,
        hypothesis_limit: int = MODEL1_MAX_HYPOTHESIS_COUNT,
        teacher_forcing_ratio: float = MODEL1_TEACHER_FORCING_RATE, # Implement if time permits
        ) -> None:
        super(Model1,self).__init__()

        # NOTE to implementer: May be useful to keep the following. Remove these if not needed in forward.
        self.vocab_size = vocab_size
        
        # Important stuff to keep for forward
        self.padding_index = padding_index
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_num_layers = decoder_num_layers
        self.start_of_summ_index = start_of_summ_index
        self.end_of_summ_index = end_of_summ_index
        self.beam_search_k = beam_search_k
        self.generation_limit = generation_limit
        self.hypothesis_limit = hypothesis_limit
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Model parts
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_index)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_p)

        self.encoder = Encoder1(
            embedding_dim=embedding_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            rnn_dropout_p=encoder_rnn_dropout_p,
            fc_dropout_p=encoder_fc_dropout_p,
            bidirectional=bidirectional_encoder,
            decoder_hidden_dim=decoder_hidden_dim,
            activation=activation
            )

        self.decoder = Decoder1(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            rnn_dropout_p=decoder_rnn_dropout_p,
            num_attention_head=decoder_num_attention_head,
            attention_dropout_p=decoder_attention_dropout_p,
            activation=activation,
            input_feeding_fc_dropout_p=decoder_input_feeding_fc_dropout_p,
            attentional_fc_out_dim=decoder_attentional_fc_out_dim
        )

        self.softmax = nn.LogSoftmax(vocab_size, dim=2)

    def forward(self, text_packed_seq: PackedSequence):
        r"""Returns best sequence decoding (with beam search) for each batch item, and the relevant logits used to compute each word token index
        """
        # Record original batch size
        N = text_packed_seq.batch_sizes[0]

        # ---------------------------------------------------------
        # -------------------- Encoder section --------------------
        # ---------------------------------------------------------

        # Embed input text sequence of word token indices
        embedded_packed_seq = embedding_apply(self.embedding_layer, text_packed_seq)
        
        # Apply dropout to the embeddings
        embedded_packed_seq = packed_seq_apply(self.embedding_dropout, embedded_packed_seq)
        
        # Push dropout-ed embeddings to encoder.
        (padded_encoder_outputs, input_seq_lengths), (decoder_initial_hidden, decoder_initial_cell) = self.encoder(embedded_packed_seq)
        # padded_encoder_outputs shape = [batch size, max input seq len, decoder hidden dim]
        # input_seq_lengths shape = [batch size]
        # decoder_initial_hidden tensor shape = [num_layers, batch size, decoder hidden dim]
        # decoder_initial_cell tensor shape = [num_layers, batch size, decoder hidden dim]

        batch_max_input_seq_len = max(input_seq_lengths).long().item()

        # Compute attention mask for parallel compute of attention across original batch
        attention_key_padding_mask = compute_attention_key_padding_mask(input_seq_lengths)
        # attention_key_padding_mask shape = [batch size, max input seq len]

        # ---------------------------------------------------------
        # -------------------- Decoder section --------------------
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Prepare start tokens by stacking appropriately
        # ---------------------------------------------------------
        decoder_input_embedded = (torch.ones(N,1) * self.start_of_summ_index).long()
        # decoder_input_embedded shape = [batch size, 1]

        # ---------------------------------------------------------
        # Embed the start tokens
        # ---------------------------------------------------------
        decoder_input_embedded = self.embedding_layer(decoder_input_embedded)
        decoder_input_embedded = self.embedding_dropout(decoder_input_embedded)
        # decoder_input_embedded shape = [batch size, 1, embedding dim]

        # ---------------------------------------------------------
        # Perform first decoding timestep
        # ---------------------------------------------------------
        decoding_timestep = 1 # used to force terminate all decoding (regardless of batch index number) when decoding_timestep hits self.generation_limit

        hidden, cell, attn_output_weights, attentional_vectors, logits = self.decoder(decoder_input_embedded, decoder_initial_hidden, decoder_initial_cell, padded_encoder_outputs, attention_key_padding_mask)
        # hidden shape = [num_layers, N, decoder_hidden_dim]
        # cell shape = [num_layers, N, decoder_hidden_dim]
        # attn_output_weights shape = [N, 1, max_source_seq_len]
        # attentional_vectors shape = [N, 1, decoder_attentional_fc_out_dim]
        # logits shape = [N, 1, vocab size]

        # ---------------------------------------------------------
        # Prepare hypothesis tracking lists
        # ---------------------------------------------------------
        completed_hypotheses = self.__init_tracking_list(N)
        stack_of_hypotheses = self.__init_tracking_list(N)

        # ---------------------------------------------------------
        # Make GeneratedWord objects from logits
        # ---------------------------------------------------------
        topk_log_prob, topk_idx = self.softmax(logits).topk(self.beam_search_k, dim=2)
        # topk_log_prob shape = [N, 1, self.beam_search_k]
        # topk_idx is LongTensor, shape = [N, 1, self.beam_search_k], 

        # iterate through row wise, to build GeneratedWords. Generate self.beam_search_k words for each of N logits.
        for batch_idx, (h, c, aw, av, l, lp, i) in enumerate(zip(hidden.permute(1,0,2), cell.permute(1,0,2), attn_output_weights, attentional_vectors, logits, topk_log_prob, topk_idx)): # permute shenanigans for hidden and cell due to iterating by batch size dimension
            h = h.permute(1,0,2)
            c = c.permute(1,0,2)
            for log_prob, idx in zip(lp.squeeze(), i.squeeze()): # has self.beam_search_k iterations here
                # log_prob is a singleton Tensor, idx is a singleton LongTensor
                if idx==self.end_of_summ_index:
                    completed_hypotheses[batch_idx].append(GeneratedWord(None, idx, l, log_prob, h, c, av, aw, batch_idx, input_seq_lengths[batch_idx], padded_encoder_outputs[batch_idx], attention_key_padding_mask[batch_idx]))
                else:
                    stack_of_hypotheses[batch_idx].append(GeneratedWord(None, idx, l, log_prob, h, c, av, aw, batch_idx, input_seq_lengths[batch_idx], padded_encoder_outputs[batch_idx], attention_key_padding_mask[batch_idx]))
        
        # For first round, we dont need to choose top-k again amongst stack_of_hypotheses
        remaining_stack_of_hypotheses = stack_of_hypotheses
        stack_of_hypotheses = self.__init_tracking_list(N)
        
        # ---------------------------------------------------------
        # Prepare isDone NumPy boolean array, to track if all N batch items are done. For use to terminate while loop.
        # ---------------------------------------------------------
        isDone = np.zeros(N)==1 # boolean array, initialize to all False
        reached_hypothesis_limit = self.__reached_hypothesis_limit(completed_hypotheses)
        isDone += reached_hypothesis_limit

        if self.generation_limit == decoding_timestep:
            isDone = np.ones(N)==1
            for batch_idx, lst in enumerate(remaining_stack_of_hypotheses):
                completed_hypotheses[batch_idx].extend(lst)
        decoding_timestep += 1

        # ---------------------------------------------------------
        # TODO (OPTIONAL): add teacher forcing implementation HERE.
        if self.teacher_forcing_ratio != None:
            assert type(self.teacher_forcing_ratio)==float and (self.teacher_forcing_ratio <= 1) and (self.teacher_forcing_ratio >= 0), f"Problem with the input teacher_forcing_ratio. Got {self.teacher_forcing_ratio}"
            raise NotImplementedError("We did not have time to implement teacher forcing yet.")
        # else:
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # while loop shenanigans
        # ---------------------------------------------------------
        while not np.all(isDone):
            batch_items_left = self.__get_notDone_indices(isDone)
            len_batch_items_left = len(batch_items_left)

            # ---------------------------------------------------------
            # Collate per batch_items_left's (we call this index `n`, `n` may be from 0 to N-1)
            # remaining hypotheses' (we call the number of remaining hypotheses `kk_n` for ~this~ batch_items_left, [], 1 <= kk_n <= k = self.beam_search_k)
            # last words (i.e. GeneratedWord's),
            # and their relevant decoder input stuff (vocab_idx, hidden, cell, padded_encoder_output, attention_key_padding_mask).
            # For each n, track kk_n -> make a list of number of last words belonging to each n. Use np.cumsum to create indices.
            # Stack all the decoder input stuff accordingly.
            # ---------------------------------------------------------
            # Prepare dummy decoder input stuff to use with torch.vstack in for loops.
            stacked_vocab = torch.empty(1, self.padding_index)
            stacked_hidden = torch.empty(1, self.decoder_num_layers, self.decoder_hidden_dim)
            stacked_cell = torch.empty(1, self.decoder_num_layers, self.decoder_hidden_dim)
            stacked_padded_encoder_output = torch.empty(1, batch_max_input_seq_len, self.decoder_hidden_dim)
            stacked_attention_key_padding_mask = torch.empty(1, batch_max_input_seq_len)

            num_hypothesis_in_stack = [0] * len_batch_items_left

            for i, batch_idx in enumerate(batch_items_left):
                for hypothesis in remaining_stack_of_hypotheses[batch_idx]:
                    stacked_vocab = torch.vstack((stacked_vocab, hypothesis.vocab_idx.unsqueeze(0))) # so that the vocab_idx tensor is of shape = [1, 1]. vstacks on dim=0
                    stacked_hidden = torch.vstack((stacked_hidden, hypothesis.hidden.permute(1,0,2)))
                    stacked_cell = torch.vstack((stacked_cell, hypothesis.cell.permute(1,0,2))) 
                    stacked_padded_encoder_output = torch.vstack((stacked_padded_encoder_output, hypothesis.padded_encoder_output))
                    stacked_attention_key_padding_mask = torch.vstack((stacked_attention_key_padding_mask, hypothesis.attention_key_padding_mask))
                    num_hypothesis_in_stack[i] += 1
            
            cum_num_hypothesis_in_stack = np.cumsum(np.array([0]+num_hypothesis_in_stack)) # will look like [0, x, x+y, ...]

            # Trim dummy entry from each decoder input stuff
            stacked_vocab = stacked_vocab[1:] # shape = [O(n*kk_n), 1]
            stacked_hidden = stacked_hidden[1:].permute(1,0,2) # shape = [decoder_num_layers, O(n*kk_n), decoder_hidden_dim]
            stacked_cell = stacked_cell[1:].permute(1,0,2) # shape = [decoder_num_layers, O(n*kk_n), decoder_hidden_dim]
            stacked_padded_encoder_output = stacked_padded_encoder_output[1:] # shape = [O(n*kk_n), batch_max_input_seq_len, decoder_hidden_dim]
            stacked_attention_key_padding_mask = stacked_attention_key_padding_mask[1:] # shape = [O(n*kk_n), batch_max_input_seq_len]

            # ---------------------------------------------------------
            # (Embed -> dropout) all vocab_idx to get embedded.
            # ---------------------------------------------------------
            decoder_input_embedded = stacked_vocab
            decoder_input_embedded = self.embedding_layer(decoder_input_embedded)
            decoder_input_embedded = self.embedding_dropout(decoder_input_embedded)

            # ---------------------------------------------------------
            # Push stack of O(n*kk_n) augmented_batch into decoder
            # ---------------------------------------------------------
            hidden, cell, attn_output_weights, attentional_vectors, logits = self.decoder(decoder_input_embedded, stacked_hidden, stacked_cell, stacked_padded_encoder_output, stacked_attention_key_padding_mask)

            topk_log_prob, topk_idx = self.softmax(logits).topk(self.beam_search_k, dim=2)

            # ---------------------------------------------------------
            # Sort outputs of decoder to relevant batch item n, and its kk_n hypothesis.
            # Use len_batch_items_left, cum_num_hypothesis_in_stack to iterate through decoder outputs.
            # ---------------------------------------------------------
            tmp_decoder_output_holder = [[] for _ in range(len_batch_items_left)]
            for i in range(len_batch_items_left):
                tmp_decoder_output_holder.append(hidden.permute(1,0,2)[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(cell.permute(1,0,2)[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(attn_output_weights[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(attentional_vectors[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(logits[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(topk_log_prob[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])
                tmp_decoder_output_holder.append(topk_idx[cum_num_hypothesis_in_stack[i]:cum_num_hypothesis_in_stack[i+1]])

            # ---------------------------------------------------------
            # For each O(n*kk_n) logit, do k-beam search, make `GeneratedWord`,
            # appending completed hypothesis to completed_hypotheses[n]
            # and appending normal hypothesis to stack_of_hypotheses[n]
            # ---------------------------------------------------------
            # NOTE: Remember to unpermute hidden and cell before saving to new `GeneratedWord`'s
            for i, batch_idx in enumerate(batch_items_left): # iterating through batch item n
                h_b, c_b, aw_b, av_b, l_b, lp_b, i_b = tmp_decoder_output_holder[i]
                for j, (h, c, aw, av, l, lp, i) in enumerate(zip(h_b, c_b, aw_b, av_b, l_b, lp_b, i_b)): # iterating through remaining hypothesis kk_n
                    h = h.permute(1,0,2)
                    c = c.permute(1,0,2)
                    for log_prob, idx in zip(lp.squeeze(), i.squeeze()): # has self.beam_search_k iterations here
                        # log_prob is a singleton Tensor, idx is a singleton LongTensor
                        if idx==self.end_of_summ_index:
                            completed_hypotheses[batch_idx].append(GeneratedWord(remaining_stack_of_hypotheses[batch_idx][j], idx, l, log_prob, h, c, av, aw))
                        else:
                            stack_of_hypotheses[batch_idx].append(GeneratedWord(remaining_stack_of_hypotheses[batch_idx][j], idx, l, log_prob, h, c, av, aw))

            # ---------------------------------------------------------
            # For each n, filter stack_of_hypotheses[n] to top-k hypothesis,
            # and replace remaining_stack_of_hypotheses[n] with stack_of_hypotheses[n].
            # Also replace remaining_stack_of_hypotheses[n] even if stack_of_hypotheses[n] is empty.
            # Reset stack_of_hypotheses.
            # ---------------------------------------------------------
            remaining_stack_of_hypotheses = self.__filter_stack_of_hypotheses(stack_of_hypotheses)
            stack_of_hypotheses = self.__init_tracking_list(N)

            # ---------------------------------------------------------
            # We check if we are done.
            # ---------------------------------------------------------
            no_more_remaining_hypothesis = self.__no_more_remaining_hypothesis(remaining_stack_of_hypotheses)
            isDone += no_more_remaining_hypothesis

            reached_hypothesis_limit = self.__reached_hypothesis_limit(completed_hypotheses)
            isDone += reached_hypothesis_limit

            if self.generation_limit == decoding_timestep:
                isDone = np.ones(N)==1
                for batch_idx, lst in enumerate(remaining_stack_of_hypotheses):
                    completed_hypotheses[batch_idx].extend(lst)
                break
            decoding_timestep += 1

            # ---------------------------------------------------------
            # Repeat while loop condition check, GO UP
            # ---------------------------------------------------------

        # ---------------------------------------------------------
        # wacky while loop over!
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Filter to best hypothesis from the N sublists of completed_hypotheses
        # ---------------------------------------------------------
        best_hypotheses = self.__get_best_hypothesis(completed_hypotheses)

        # ---------------------------------------------------------
        # Get index sequence, logits and attention weights from best hypothesis of each original batch of N!
        # ---------------------------------------------------------
        output_seq = list(map(lambda hypothesis: hypothesis.get_hypothesis_word_indices(), best_hypotheses))
        # output_seq is a list of N lists, with each inner list holding the indices of the generated summary
        output_logits = list(map(lambda hypothesis: hypothesis.get_hypothesis_logits(), best_hypotheses))
        # output_logits is a list of N tensors, with each tensor having shape = [generated seq len, vocab size]
        attn_weights = list(map(lambda hypothesis: hypothesis.get_attn_weights(), best_hypotheses))
        # attn_weights is a list of N tensors, with each tensor having shape = [generated seq len, batch_max_input_seq_len]

        # TODO Consider what to do if generated summary is shorter than GT summary, to compute loss.
        # Better to do in train.py and test.py
        return output_seq, output_logits, attn_weights

    def __init_tracking_list(self, N:int) -> list:
        return list([] for _ in range(N))
    
    def __compute_lengths_of_inner_list(self, l:list) -> list:
        r"""Internal helper function to map list of lists to list of lengths of inner lists.
        """
        return list(map(lambda lst: len(lst), l))
    
    def __reached_hypothesis_limit(self, l:list) -> np.ndarray:
        r"""For use with `completed_hypotheses` list.
        
        Returns NumPy boolean array.
        """
        return np.array(self.__compute_lengths_of_inner_list(l)) >= self.hypothesis_limit
    
    def __get_notDone_indices(self, isDone:np.array) -> list:
        r"""Internal helper function to get list of indices where isDone is False."""
        return np.argwhere(isDone==False).squeeze().tolist()
    
    def __filter_stack_of_hypotheses(self, stack_of_hypotheses):
        r"""Internal helper function to select top-k from O(k^2) hypotheses in input `stack_of_hypotheses`."""
        return list(map(lambda lst: sorted(lst, reverse=True)[:self.beam_search_k] if lst!=[] else [], stack_of_hypotheses))
    
    def __no_more_remaining_hypothesis(self, l:list) -> list:
        r"""For use with `remaining_stack_of_hypotheses` list.
        
        Returns NumPy boolean array.
        """
        return np.array(self.__compute_lengths_of_inner_list(l)) == 0
    
    def __get_best_hypothesis(self, l:list):
        r"""For use with `completed_hypotheses` list.
        
        Returns list of best hypothesis
        """
        return list(map(lambda lst: sorted(lst,reverse=True)[0], l))