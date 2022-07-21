from torch import nn


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
