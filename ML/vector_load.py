import torch
import numpy as np

from inference import infer, load_for_inference
from dataloader import SequenceDataset, load_sequences

sequence, label = load_sequences('trainingdata')

model, tokenizer, label_dict = load_for_inference('emb.pth','gene_tokenizer.json','label_dict.json',
 skip_gram=True)

batch_size = 1000
num_batches = len(sequence) // batch_size
if len(sequence) % batch_size != 0:
    num_batches + 1

all_embeddings = []
for i in range(num_batches):
    emb = infer(sequence[i * batch_size: (i+1) * batch_size], tokenizer, model)
    all_embeddings.append(emb)

all_embeddings = torch.cat(all_embeddings, 0).numpy()

np.save('virus_embeddings.npy')

#PCA

from sklearn.decomposition import PCA
import pickle

PCA(n_components=3) # reduce to 3d
pca.fit(all_embeddings)
embeddings_3d = pca.transform(all_embeddings)

np.save('virus_embeddings_3d.npy',embeddings_3d)

with open('virus_pca.pkl','wb') as f:
    pickle.dump(f,pca)