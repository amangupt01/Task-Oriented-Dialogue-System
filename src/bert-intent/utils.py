from imports import np, pd, jsonlines, torch, \
    TensorDataset, DataLoader, RandomSampler, SequentialSampler
from constants import params

def get_intent_dict(file_path):
    intent_dict = {}
    intent_counter = 0
    with jsonlines.open(file_path) as reader: 
        for obj in reader: 
            output_raw = obj['output'] 
            intent = output_raw.split(' ')[0]
            if intent not in intent_dict:
                intent_dict[intent] = intent_counter
                intent_counter += 1
    return intent_dict

def create_dataset(file_path, intent_dict):
    texts = []
    labels = []
    with jsonlines.open(file_path) as reader: 
        for obj in reader: 
            input_raw = obj['input'] 
            output_raw = obj['output'] 
            intent = output_raw.split(' ')[0]
            texts.append(input_raw)
            labels.append(intent_dict[intent])
    # make labels as class int
    labels = np.array(labels, dtype=np.int64)
    return pd.DataFrame({'text': texts, 'label': labels})

def get_tokens(text, tokenizer):
    tokens = tokenizer.batch_encode_plus(
        text.tolist(),
        max_length = params["max_seq_len"],
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )
    return tokens

def to_tensors(data, labels):
    return (torch.tensor(data['input_ids'], dtype=torch.long),
            torch.tensor(data['attention_mask'], dtype=torch.long),
            torch.tensor(labels.tolist(), dtype=torch.long))

def get_dataloader(seq, mask, y, type_):
    # wrap tensors
    data = TensorDataset(seq, mask, y)
    # sampler for sampling the data during training
    if type_ == 'train':
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    # dataLoader for train set
    dataloader = DataLoader(data, sampler=sampler, batch_size=params["batch_size"])
    return dataloader