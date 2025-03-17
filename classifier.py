import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import Dataloader
from proformer import TransformerModel
from params import bpi_params
from run_proformer import parse_params
# Initialize loader, model, and opt
print("-- PARSING CMD ARGS --")
opt = parse_params()
for k in bpi_params.keys():
    opt[k] = bpi_params[k]
print(opt)
loader = Dataloader(filename=opt["dataset"], opt=opt)
model = TransformerModel(len(loader.vocab), opt).to(opt["device"])

# Carica il vocabolario e i dataset
vocab, dtrain, dval, dtest = loader.get_dataset(num_test_ex=1000)
cls1, cls2 = vocab.get_stoi()["1"], vocab.get_stoi()["2"]

# Recupera un batch di esempi
data, targets = loader.get_batch(loader.test_data, 0)
attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

# Inferenza sul modello pre-trained
with torch.no_grad():
    output = model(data, attn_mask)
    output_flat = output.view(-1, model.ntokens)

# Crea una maschera dove i target sono equivalenti al token classe
cls_mask = (targets == cls1) | (targets == cls2)
targets = targets[cls_mask]
output_flat = output_flat[cls_mask, :]

# Salva gli embedding relativi alla classe
class_embeddings = model.last_hidden_states.view(-1, 64)[cls_mask].view(-1, 1, 64)

# Definisci un semplice MLP per la classificazione
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Ipotizziamo che gli embedding abbiano dimensione 64
input_dim = 64
hidden_dim = 32
output_dim = 2  # Numero di classi

# Inizializza il classificatore
classifier = MLPClassifier(input_dim, hidden_dim, output_dim).to(opt["device"])

# Definisci l'ottimizzatore e la funzione di perdita
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Esempio di training loop per il classificatore
for epoch in range(10):  # Numero di epoche
    classifier.train()
    optimizer.zero_grad()
    outputs = classifier(class_embeddings)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Inferenza con il classificatore addestrato
classifier.eval()
with torch.no_grad():
    outputs = classifier(class_embeddings)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted classes: {predicted}')