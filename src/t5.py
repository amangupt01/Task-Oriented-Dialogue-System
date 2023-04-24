from imports import *
from evaluations import Evaluator
from transformers import T5Tokenizer, T5ForConditionalGeneration
import wandb

class T5_Dataset(Dataset):
    def __init__(self, data_paths, tokenizer, aux_input=False, val=False):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.input_attention_masks = []
        self.output_ids = []
        self.output_attention_masks = []
        for data_path in data_paths:
            print('Loading data from {}'.format(data_path))
            with open(data_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    obj = json.loads(line.strip())
                    input_raw = self.get_input(obj) if not aux_input else self.get_aux_input(obj)
                    output_raw = self.get_output(obj) if not val else ''
                    input_txt = "parse: " + input_raw.strip()
                    encoded_input = tokenizer(input_txt, return_tensors='pt', return_attention_mask=True)
                    self.input_ids.append(encoded_input['input_ids'][0])
                    self.input_attention_masks.append(encoded_input['attention_mask'][0])
                    output_txt = output_raw.strip()
                    encoded_output = tokenizer(output_txt, return_tensors='pt', return_attention_mask=True)
                    self.output_ids.append(encoded_output['input_ids'][0])
                    self.output_attention_masks.append(encoded_output['attention_mask'][0])
        print("Number of examples: {}".format(len(self.input_ids))) 
    def __len__(self):

        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'input_attention_mask': self.input_attention_masks[idx], 'output_ids': self.output_ids[idx], 'output_attention_mask': self.output_attention_masks[idx]}
    
    def get_input(self, obj):
        input_str = obj['input']
        return input_str

    def get_aux_input(self, obj):
        input_str = obj['input']
        try:
            history_str = ' '.join([x['user_query'] + ' ' + x['response_text'] for x in obj['history']])
        except:
            history_str = ''
        try:
            user_lists_str = ' '.join([l['name'] + ' ' + ' '.join(l['items']) for l in obj['user_lists']])
        except:
            user_lists_str = ''
        try:
            user_notes_str = ' '.join([n['name'] + ' ' + n['content'] for n in obj['user_notes']])
        except:
            user_notes_str = ''
        try:
            user_contacts_str = ' '.join(obj['user_contacts'])
        except:
            user_contacts_str = ''
        return input_str + ' <user_contacts> ' + user_contacts_str.strip() + ' <history> ' + history_str.strip() + ' <user_list> ' + user_lists_str.strip() + ' <user_notes> ' + user_notes_str.strip()
        # return input_str + ' <user_contacts> ' +  user_contacts_str.strip()
    def get_output(self, obj):
        output_str = obj['output']
        return output_str


class T5_TOD():
    def __init__(self, args):

        if (args['save_wandb']):
            wandb.login()
            wandb.init(
                project="task-oriented-dialogue-system",
                config=args
            )

        model = T5ForConditionalGeneration.from_pretrained(args['model'])
        tokenizer = T5Tokenizer.from_pretrained(args['model'])
        self.model = model.to(device)
        self.tokenizer = tokenizer
        print("Loading dataset...")
        self.train_dataset = T5_Dataset([args['train_path']], tokenizer, aux_input=args['aux_input'])
        self.dev_dataset = T5_Dataset([args['dev_path']], tokenizer, val=True, aux_input=args['aux_input'])
        self.args = args
        self.create_dataloaders(args)

        if args['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=args['init_lr'], weight_decay=0.01)
        if (args['scheduler'] == 'linear_warmup'):
            num_training_steps = args['num_epochs'] * len(self.train_data)
            num_warmup_steps = int(args['warmup_steps'] * num_training_steps)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
          
        self.evaluator = Evaluator(args['dev_path'])

    def create_dataloaders(self, args):
        self.train_data = DataLoader(self.train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=self.collate_fn)
        print("Loaded Train Dataset with {} batches".format(math.ceil(len(self.train_dataset) / args['batch_size'])))
        self.dev_data = DataLoader(self.dev_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=self.collate_fn)
        print("Loaded Dev Dataset with {} batches".format(math.ceil(len(self.dev_dataset) / args['batch_size'])))
    
    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence([x['input_attention_mask'] for x in batch], batch_first=True, padding_value=0)
        output_ids = torch.nn.utils.rnn.pad_sequence([x['output_ids'] for x in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        output_attention_mask = torch.nn.utils.rnn.pad_sequence([x['output_attention_mask'] for x in batch], batch_first=True, padding_value=0)
        return {'input_ids': input_ids, 'input_attention_mask': attention_mask, 'output_ids': output_ids, 'output_attention_mask': output_attention_mask}
    

    def train_epoch(self, epoch):
        self.model.train()
        num_batches = len(self.train_data)
        running_loss = 0
        self.optimizer.zero_grad()
        j = 0
        with tqdm(total=num_batches, desc="Training", unit="batch", leave=False) as pbar:
            for _, batch in enumerate(self.train_data):
                input_ids = batch['input_ids'].to(device)
                input_attention_mask = batch['input_attention_mask'].to(device)
                output_ids = batch['output_ids'].to(device)
                output_attention_mask = batch['output_attention_mask'].to(device)
                output_ids[output_ids == self.tokenizer.pad_token_id] = -100
                outputs = self.model(input_ids, attention_mask = input_attention_mask, labels=output_ids)
                loss = outputs[0]
                running_loss += loss.item()
                loss = loss / self.args['grad_accum_steps']
                loss.backward()
                j+=1
                if (j % self.args['grad_accum_steps'] == 0):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                pbar.set_description(f"Epoch: {epoch+1}")
                pbar.update(1)
            if (j % self.args['grad_accum_steps'] == 0):
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            print(f"Training loss: {running_loss / num_batches:.4f}")
            pbar.close()

    def train(self, args):
        print("Training started ...")
        best_so_far = 0
        for epoch in range(args['num_epochs']):
            self.train_epoch(epoch)
            self.model.eval()
            predictions = []
            if ((epoch+1)%args['eval_after'] == 0):
                outfile = open(args['output_path'], 'w')
                with torch.no_grad():
                    for batch in self.dev_data:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['input_attention_mask'].to(device)
                        outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, do_sample=True, top_p=0.95, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
                        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        predictions.extend(outputs)
                metrics = self.evaluator.compute_metrics(predictions)   
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}", end=', ')
                print()      
                if (args['save_wandb']):
                    wandb.log(metrics)
                if (metrics['accuracy'] > best_so_far):
                    best_so_far = metrics['accuracy']
                    outfile.write('\n'.join(predictions))
