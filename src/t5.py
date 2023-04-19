from imports import *
from evaluations import Evaluator
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb

class T5_Dataset(Dataset):
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
        return input_str + ' <user_contacts> ' +  user_contacts_str.strip()
    def get_output(self, obj):
        output_str = obj['output']
        return output_str


class GPT2_TOD():
    def __init__(self, args):

        if (args['save_wandb']):
            wandb.login()
            wandb.init(
                project="task-oriented-dialogue-system-gpt2",

                config={
                    "init_lr": args['init_lr'],
                    "batch_size": args['batch_size'],
                    "optimizer": args['optimizer'],
                    "num_epochs": args['num_epochs'],
                    "warmup_steps": args['warmup_steps']
                }
            )

        model = T5ForConditionalGeneration.from_pretrained('gpt2')
        tokenizer = T5Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '<pad>', 
                                        'bos_token': '<bos>',
                                        'eos_token': '<eos>'})
        tokenizer.add_tokens(['<sep>', '<history>', '<user_list>', '<user_notes>', '<user_contacts>'])
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = T5_Dataset([args['train_path']], tokenizer, aux_input=False)
        self.dev_dataset = T5_Dataset([args['dev_path']], tokenizer, val=True, aux_input=False)
    
        self.create_dataloaders(args)

        if args['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=args['init_lr'], weight_decay=0.01)
        if (args['scheduler'] == 'linear_warmup'):
            num_training_steps = args['num_epochs'] * len(self.train_data)
            num_warmup_steps = int(0.1 * num_training_steps)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
       
        self.evaluator = Evaluator(args['dev_path'])

    def create_dataloaders(self, args):
        self.train_data = DataLoader(self.train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=self.collate_fn)
        print("Loaded Train Dataset with {} batches".format(math.ceil(len(self.train_dataset) / args['batch_size'])))
        self.dev_data = DataLoader(self.dev_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=self.collate_fn)
        print("Loaded Dev Dataset with {} batches".format(math.ceil(len(self.dev_dataset) / args['batch_size'])))
    
    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    

    def train_epoch(self, epoch):
        self.model.train()
        num_batches = len(self.train_data)
        running_loss = 0
        with tqdm(total=num_batches, desc="Training", unit="batch", leave=False) as pbar:
            for _, batch in enumerate(self.train_data):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.model(input_ids, attention_mask = attention_mask, labels=input_ids)
                loss = outputs[0]
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_description(f"Epoch: {epoch+1}")
                pbar.update(1)
            print(f"Training loss: {running_loss / num_batches:.4f}")

    def train(self, args):
        print("Training started ...")
        best_so_far = 0
        for epoch in range(args['num_epochs']):
            self.train_epoch(epoch)
            self.model.eval()
            predictions = []
            outfile = open(args['output_path'], 'w')
            with torch.no_grad():
                for batch in self.dev_data:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=40, do_sample=True,top_p=0.95, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
                    outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    outputs = [x.split('<sep>')[1].strip() for x in outputs]
                    predictions.extend(outputs)
            outfile.write('\n'.join(predictions))
            metrics = self.evaluator.compute_metrics(predictions)   
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}", end=', ')
            print()      
            if (args['save_wandb']):
                wandb.log(metrics)
            if (metrics['accuracy'] > best_so_far):
                best_so_far = metrics['accuracy']
                torch.save(self.model.state_dict(), args['model_path'])
        