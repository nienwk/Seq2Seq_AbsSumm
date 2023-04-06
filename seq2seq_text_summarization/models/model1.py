import torch.nn as nn

class model1(nn.Module):
    def __init__(self, embedding_dim, encoder, decoder):
        super(model1,self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, text_packed_seq, summ_packed_seq):
        pass
