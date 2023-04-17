from imports import pickle, os, torch,\
    AdamW, compute_class_weight, np, nn,\
    tqdm
from imports import AutoModel, BertTokenizerFast
from utils import get_intent_dict, create_dataset, get_tokens\
    , to_tensors, get_dataloader
from constants import params
from model import BERT_Arch

device = torch.device("cuda")
data_dir = "../../data/"


train_data_path = os.path.join(data_dir, "train.jsonl")
val_data_path = os.path.join(data_dir, "dev.jsonl")

# Get the intent dictionary
intent_pickle_path = os.path.join(data_dir, "intent_dict.pickle")
if os.path.exists(intent_pickle_path):
    with open(intent_pickle_path, 'rb') as handle:
        intent_dict = pickle.load(handle)
else:
    intent_dict = get_intent_dict(train_data_path)
    with open(intent_pickle_path, 'wb') as handle:
        pickle.dump(intent_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create the train and validation datasets
train_df = create_dataset(train_data_path, intent_dict)
val_df = create_dataset(val_data_path, intent_dict)


train_text = train_df['text']
train_labels = train_df['label']
val_text = val_df['text']
val_labels = val_df['label']
bert = AutoModel.from_pretrained('bert-base-uncased')
for param in bert.parameters():
    param.requires_grad = False
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokens_train = get_tokens(train_text, tokenizer)
tokens_val = get_tokens(val_text, tokenizer)

train_seq, train_mask, train_y = to_tensors(tokens_train, train_labels)
val_seq, val_mask, val_y = to_tensors(tokens_val, val_labels)

train_dataloader = get_dataloader(train_seq, train_mask, train_y, 'train')
val_dataloader = get_dataloader(val_seq, val_mask, val_y, 'val')

model = BERT_Arch(bert)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr = params['lr'])
class_wts = compute_class_weight(class_weight='balanced', classes = np.unique(train_labels),y= train_labels)
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
cross_entropy  = nn.NLLLoss(weight=weights) 
epochs = params['epochs']

# function to train the model
def train():
    model.train()
    total_loss = 0
    for _,batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
    
        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)

        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate():
    model.eval()
    predictions = []
    gt = []
    for _,batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            preds = model(sent_id, mask)
            preds = preds.detach().cpu().numpy()
        predictions.append(preds)
        gt.append(labels.cpu().numpy())
    predictions  = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis = 1)
    gt = np.concatenate(gt, axis=0)
    accuracy = np.sum(predictions == gt)/len(gt)
    return accuracy


# set initial loss to infinite
best_valid_accuracy = 0
#for each epoch
for epoch in range(params['epochs']):
    
    #train model
    train_loss = train()
    
    #evaluate model
    validation_accuracy = evaluate()
    
    #save the best model
    if validation_accuracy > best_valid_accuracy:
        best_valid_accuracy = validation_accuracy
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print(f'\n Epoch {epoch} / {params["epochs"]} -> Training Loss {train_loss} || Accuracy:{validation_accuracy}')
    