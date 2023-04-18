from imports import *

class TODDataset(Dataset):
    def __init__(self, data_paths, tokenizer, aux_input=False, val=False):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attention_mask = []
        for data_path in data_paths:
            print('Loading data from {}'.format(data_path))
            with jsonlines.open(data_path) as reader:
                for obj in reader:
                    input_raw = self.get_input(obj) if not aux_input else self.get_aux_input(obj)
                    output_raw = self.get_output(obj) if not val else ''
                    text = "<bos> " + input_raw.strip() + " <sep> " + output_raw.strip() + " <eos>"
                    encoded_input = tokenizer(text, return_tensors='pt', return_attention_mask=True)
                    self.input_ids.append(encoded_input['input_ids'][0])
                    self.attention_mask.append(encoded_input['attention_mask'][0])
        print("Number of examples: {}".format(len(self.input_ids))) 
    def __len__(self):

        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
    
    def get_input(self, obj):
        input_str = obj['input']
        return input_str

    def get_aux_input(self, obj):
        input_str = obj['input']
        history_str = ' '.join([x['user_query'] + ' ' + x['response_text'] for x in obj['history']])
        user_lists_str = ' '.join([l['name'] + ' ' + ' '.join(l['items']) for l in obj['user_lists']])
        user_notes_str = ' '.join([n['name'] + ' ' + n['content'] for n in obj['user_notes']])
        user_contacts_str = ' '.join(obj['user_contacts'])
        # return input_str + ' <history> ' + history_str.strip() + ' <user_list> ' + user_lists_str.strip() + ' <user_notes> ' + user_notes_str.strip() + ' <user_contacts> ' + user_contacts_str.strip()
        return input_str + ' <history> ' + history_str.strip()
    def get_output(self, obj):
        output_str = obj['output']
        return output_str
    
def collate_fn(batch):
    return
    # Group samples by length
    batch = sorted(batch, key=lambda x: x['input_ids'].size(0))
    
    # Get input_ids and attention_mask tensors
    input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}