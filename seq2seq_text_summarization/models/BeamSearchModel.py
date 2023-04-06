import torch
import torch.nn as nn

class BeamSearchSummarizationModel(nn.Module):
    # Vocab Size: Size of vocab of nlp model
    # Embedding Dim: May Vary - Size for embedding layer
    # Hidden Dim: May Vary - Size for Hidden Layer
    # enc_num_layers: Number of Encoder Stacked LSTM
    # dec_num_attn_head: Number of heads for multihead attention layer in decoder
    # Beam Size: Beam Width of Beam Search Algo
    # output_max_len: Max Length of generated summary, output will be (output_max_len, Batch Size)
    def __init__(self, vocab_size, embedding_dim, hidden_dim, enc_num_layers, dec_num_attn_head = 1, beam_size = 3, output_max_len = 100):
        self.beam_size = beam_size
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size
        self.encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, enc_num_layers)
        self.decoder = LSTMDecoderWithAttention(vocab_size, embedding_dim, hidden_dim, dec_num_attn_head)
    
    def forward(self, text):
        # Assume text is numericalize -> each word represented by an index
        # text = (Sentence Length, batch size)
        
        batch_size = text.size(1)
        
        # enc_hidden_n = (enc_num_layers, Batch Size, Hidden Dim)
        # enc_cell_n = (enc_num_layers, Batch Size, Hidden Dim)
        enc_hidden_n, enc_cell_n = self.encoder(text)
        
        last_enc_hidden_n = enc_hidden_n[-1::] # (1, Batch Size, Hidden Dim)
        last_enc_cell_n = enc_cell_n[-1::]     # (1, Batch Size, Hidden Dim)
        
        list_last_hidden_n = []
        list_last_cell_n = []
        for index_in_batch in range(self.beam_size):
            list_last_hidden_n.append(last_enc_hidden_n)
            list_last_cell_n.append(last_enc_cell_n)
        
        
        # result (Beam Size, Sentence Length , Batch Size)
        # Init with <sos>
        result = torch.full((self.beam_size, 1, batch_size), 1)
        
        # logits (Beam Size, Sentence Length, Batch Size, Vocab Size)
        # Init for <sos>
        logits = torch.full((self.beam_size, 1, batch_size, self.vocab_size), 0)
        for beam_index in range(self.beam_size):
            for batch_index in range(batch_size):
                logits[beam_index][0][batch_index][1] = 1
        
        
        for sentence_index in range(self.output_max_len):
            list_dec_hidden_n = []
            list_dec_cell_n = []
            list_word_probs = []
            list_word_token = []
            list_dec_output = []
            
            prev_result = result.clone() # Save result from previous loop
            prev_logits = logits.clone() # Save logits from previous loop
            
            for beam_index in range(self.beam_size):
                # Take previous word for decoder input
                dec_input = result[beam_index][-1:] # (1, Batch Size)
                last_hidden_n = list_last_hidden_n[beam_index] # (Batch Size, Hidden Dim)
                last_cell_n = list_last_cell_n[beam_index] # (Batch Size, Hidden Dim)
                
                # Input to Decoder
                # dec_output = (Batch Size, Vocab Size)
                # dec_hidden_n = (1, Batch Size, Hidden Dim)
                # dec_cell_n = (1, Batch Size, Hidden Dim)
                dec_output, dec_hidden_n, dec_cell_n = self.decoder(dec_input, last_hidden_n, last_cell_n)
                
                list_dec_output.append(dec_output)
                
                # Save dec_hidden_n and dec_cell_n
                list_dec_hidden_n.append(dec_hidden_n)
                list_dec_cell_n.append(dec_cell_n)
                
                # Get Top K of Current Beam
                # word_prob = (Batch Size, Beam Size)
                # word_token = (Batch Size, Beam Size)
                word_prob, word_token = torch.topk(dec_output, self.beam_size, dim=1)
                list_word_probs.add(word_prob)
                list_word_token.add(word_token)
            
            # Top (Beam Size * Beam Size) Word probailities and tokens
            all_top_word_prob = torch.concat(list_word_probs, dim = 1) # (Batch Size, Beam Size * Beam Size)
            all_top_word_token = torch.concat(list_word_token, dim = 1) # (Batch Size, Beam Size * Beam Size)
            
            # Get Top K from all beam output
            # top_k_val = (Batch Size, Beam Size)
            # indices_of_top_word_prob = (Batch Size, Beam Size)
            top_k_val, indices_of_top_word_prob = torch.topk(all_top_word_prob, self.beam_size, dim=1) 
            
            # top_k_word_token = (Batch Size, Beam Size) 
            # Each entry top_k_word_token[i] should be the top K words of that entry
            top_k_word_token = torch.zeros_like(indices_of_top_word_prob)
            
            # Sort Data According to Top K Words
            list_last_hidden_n = []
            list_last_cell_n = []
            list_logits_to_add = []
            for index_in_batch in range(indices_of_top_word_prob.size(dim=0)):
                for new_beam_index in range(indices_of_top_word_prob.size(dim=1)):
                    index = indices_of_top_word_prob[index_in_batch][new_beam_index]
                    
                    # Get beam index based on old result tensor
                    old_beam_index = index / self.beam_size
                    
                    # Update result tensor for existing sentence fragment
                    result, logits = self.copy_existing_sentence_frag(result, prev_result, logits, prev_logits, index_in_batch, new_beam_index, old_beam_index)
                    
                    # Save relevant hidden_n, cell_n and logits output from decoder
                    list_last_hidden_n.append(list_dec_hidden_n[old_beam_index])
                    list_last_cell_n.append(list_dec_cell_n[old_beam_index])
                    list_logits_to_add.append(list_dec_output[old_beam_index])
                    
                    # Get word token
                    word_token = all_top_word_token[index_in_batch][index]
                    top_k_word_token[index_in_batch][new_beam_index] = word_token
            
            
            # Concat new word to result tensor
            tmp = top_k_word_token.permute(1,0).unsqueeze(1) # (Beam Size, 1, Batch Size)
            result = torch.concat((result,tmp), 1)
            
            # Concat new logits to tensor
            tmp_logits_stack = torch.stack(list_logits_to_add).unsqueeze(1) # (Beam Size, 1, Batch Size, Vocab Size)
            logits = torch.concat((logits,tmp_logits_stack), 1)
            
        
        
        # Output Result (Batch Size, output_max_len)
        output_res = result[0].permute(1, 0)
        
        # Output Logits (Batch Size, Vocab Size, output_max_len)
        output_logits = logits[0].permute(1,2,0)
        
        return  output_res, output_logits
    
    # Copy the existing sentence fragment from results-tensor of previous decoder-loop to a different beam index in the current result-tensor
    def copy_existing_sentence_frag(self, curr_result, prev_result, curr_logits, prev_logits, index_in_batch, new_beam_index, old_beam_index):
        result = curr_result.clone()
        logits = curr_logits.clone()
        
        # result (Beam Size , Sentence Length , Batch Size)
        # logits (Beam Size, Sentence Length, Batch Size, Vocab Size)
        beam_size = self.beam_size
        for sentence_index in range(prev_result.size(dim=1)):
            result[new_beam_index][sentence_index][index_in_batch] = prev_result[old_beam_index][sentence_index][index_in_batch]
            logits[new_beam_index][sentence_index][index_in_batch] = prev_logits[old_beam_index][sentence_index][index_in_batch]

        return result, logits


# --- Encoder ---
class LSTMEncoder(nn.Module):
    # Vocab Size: Size of vocab of nlp model
    # Embedding Dim: May Vary - Size for embedding layer
    # Hidden Dim: May Vary - Size for Hidden Layer
    # Num Layers: Number of stacked LSTM
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        
        super().__init__()
        
        # Assume randomized embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Setup
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers)
        
        
        
    def forward(self, text):
        # Assume text is numericalize -> each word represented by an index
        # text = (Sentence Length, batch size)

        #embedded = (Sentence Length, Batch Size, Embedding Dim)
        embedded = self.embedding(text)
        
        # hidden_n = (num_layers, Batch Size, Hidden Dim)
        # cell_n = (num_layers, Batch Size, Hidden Dim)
        output, (hidden_n, cell_n) = self.lstm(embedded)
        
        return hidden_n, cell_n


# --- Decoder ---
class LSTMDecoderWithAttention(nn.Module):
    # Vocab Size: Size of vocab of nlp model
    # Embedding Dim: May Vary - Size for embedding layer
    # Hidden Dim: May Vary - Size for Hidden Layer
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_attn_head = 1):
        
        super().__init__()
        
        # Assume randomized embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM Setup
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        # Attention Setup
        self.attn = nn.MULTIHEADATTENTION(hidden_dim, num_attn_head)
        
        # Linear Layer (Output = Vocab Size)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        
        
    def forward(self, text, last_hidden_n, last_cell_n, enc_hidden_n):
        # Assume text is numericalize -> each word represented by an index
        # Assume text to be one word only
        # text = (1, batch size)
        # last_hidden_n = (1, Batch Size, Hidden Dim)
        # last_cell_n = (1, Batch Size, Hidden Dim)
        # enc_hidden_n = (Num Layers of Encoder, Batch Size, Hidden Dim)
        

        #embedded = (1, Batch Size, Embedding Dim)
        embedded = self.embedding(text)
        
        
        # dec_hidden_n = (1, Batch Size, Hidden Dim)
        # dec_cell_n = (1, Batch Size, Hidden Dim)
        _, (dec_hidden_n, dec_cell_n) = self.lstm(embedded, (last_hidden_n, last_cell_n))
        
        # Stack all lstm hidden layers for input to attention layer
        # all_hidden_n = (Num Layers of Encoder + 1, Batch Size, Hidden Dim)
        all_hidden_n = torch.concat((enc_hidden_n, dec_hidden_n))
        
        # attn_output = (1, Batch Size, Hidden Dim)
        attn_output, _ = self.attn(query = dec_hidden_n, key = all_hidden_n, value = all_hidden_n)
        
        # output = (Batch Size, Vocab Size)
        output = self.linear(attn_output.squeeze(0))
        
        
        return output, dec_hidden_n, dec_cell_n

