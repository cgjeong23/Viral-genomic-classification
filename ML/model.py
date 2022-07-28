from torch import nn
import torch

class SkipGramEmbeddingModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, pad_id, window_size):
        super().__init__()
        self.pad_id = pad_id
        emb_weight = torch.rand([vocab_size, embedding_dim])
        emb_weight[pad_id] = 0
        self.embedding = nn.Embedding.from_pretrained(emb_weight,
        freeze=False, padding_idx=pad_id)
        self.outlayer = nn.Linear(embedding_dim, vocab_size)
        self.window_size = window_size
   
    def forward(self, ids):
        embedded_ids = self.embedding(ids) # [B,L E]
        grouped_emb = []
        for i in range(ids.shape[1]):
            if i - self.window_size < 0:
                group = embedded_ids[:, 1: i + self.window_size].sum(1)
            elif i + self.window_size > ids.shape[1]:
                group = embedded_ids[:, i-self.window_size:-1].sum(1)
            else:
                group = embedded_ids[:, i-self.window_size:i+self.window_size].sum(1)
                group -= embedded_ids[:,i]

            grouped_emb.append(group) # [ [B,E] ]

        grouped_emb = torch.stack(grouped_emb, 1) #[B,L,E]
        out = self.out_layer() # [B, L, V]
        return out

class RnnModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pad_id, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids):
        # ids: [batch size, max sequence length] = [B, L]
        embedded = self.embedding(ids)  # [B, L, E]
        rnn_out, _ = self.rnn(embedded)  # [B, L, H]
        return self.out_layer(rnn_out)  # [B, L, V]


class RnnModelForClassification(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pad_id, hidden_dim, num_layers,
                 output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.out_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, ids):
        # ids: [batch size, max sequence length] = [B, L]
        embedded = self.embedding(ids)  # [B, L, E]
        rnn_out, _ = self.rnn(embedded)  # [B, L, H]
        hidden_state = rnn_out[:, -1, :]  # [B, H]
        return self.out_layer(hidden_state)  # [B, C]
