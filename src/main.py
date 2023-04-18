
from imports import *

from dataloader import TODDataset
from utils import *
from evaluations import Evaluator
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def train_epoch(epoch, model, optimizer, scheduler, train_data):
    model.train()
    num_batches = len(train_data)
    running_loss = 0
    with tqdm(total=num_batches, desc="Training", unit="batch", leave=False) as pbar:
        for _, batch in enumerate(train_data):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask, labels=input_ids)
            loss = outputs[0]
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Epoch: {epoch}")
            pbar.update(1)
        print(f"Training loss: {running_loss / num_batches:.4f}")
        torch.save(model.state_dict(), args['model_path'])

def train(model, tokenizer, optimizer, scheduler, train_data, dev_data, evaluator, args):
    print("Training started ...")
    for epoch in range(args['num_epochs']):
        train_epoch(epoch, model, optimizer, scheduler, train_data)
        model.eval()
        predictions = []
        outfile = open(args['output_path'], 'w')
        with torch.no_grad():
            for batch in dev_data:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=40, do_sample=True,top_p=0.95, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                outputs = [x.split('<sep>')[1].strip() for x in outputs]
                predictions.extend(outputs)
        outfile.write('\n'.join(predictions))
        metrics = evaluator.compute_metrics(predictions)   
        print(metrics) 
        # for k, v in metrics.items():
        #     print(f"{k}: {v:.4f}", end=', ')
        # print()      

if __name__ == '__main__':
    seed_everything(42)
    parser = ArgumentParser().parse_args()
    args = vars(parser)
    print(args)

    if (args['model'].startswith('gpt2')):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '<pad>', 
                                        'bos_token': '<bos>',
                                        'eos_token': '<eos>'})
        tokenizer.add_tokens(['<sep>', '<history>', '<user_list>', '<user_notes>', '<user_contacts>'])
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
    if (not args['use_random_split']):
        train_dataset = TODDataset([args['train_path']], tokenizer, aux_input=False)
        dev_dataset = TODDataset([args['dev_path']], tokenizer, val=True, aux_input=False)

    def collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([x['input_ids'] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence([x['attention_mask'] for x in batch], batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_data = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    print("Loaded Train Dataset with {} batches".format(len(train_dataset)))
    dev_data = DataLoader(dev_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
    print("Loaded Dev Dataset with {} batches".format(len(dev_dataset)))
    
    num_training_steps = args['num_epochs'] * len(train_data)
    num_warmup_steps = int(0.1 * num_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['init_lr'], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    evaluator = Evaluator(args['dev_path'])
    train(model, tokenizer, optimizer, scheduler, train_data, dev_data, evaluator, args)
    # for batch in dev_data:
    #     input_ids = batch['input_ids'].to(device)
    #     for x in tokenizer.batch_decode(input_ids, skip_special_tokens=True):
    #         print(x)