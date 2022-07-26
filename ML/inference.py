import torch
from tokenizers import Tokenizer, models
from model import RnnModelForClassification

def infer(sequence, tokenizer, model):
    encoded_sequence = tokenizer.encode(sequence)
    encoded_ids = torch.LongTensor()(encoded_sequence.ids).unsqueeze(0) # [1, L]

    pred = model(encoded_ids).squeeze(0) # [C,]
    prob = torch.softmax(pred,0).cpu().numpy()

    return prob

def load_for_inference(model_path, tokenizer_file, embedding_dim=256, hidden_dim=512,
 num_layers=1):
    tokenizer = Tokenizer(models.BPE())
    tokenizer = tokenizer.from_file(tokenizer_file)

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.padding['pad_id']
    model = RnnModelForClassification(vocab_size, embedding_dim, pad_id, hidden_dim,
     num_layers, 6)
    
    model.load_state_dict(torch.load(model_path))

    return model, tokenizer
    
