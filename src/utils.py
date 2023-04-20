from imports import *
from dataloader import DataLoader

class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', type=str, default='data/train.jsonl')
        parser.add_argument('--dev_path', type=str, default='data/dev.jsonl')
        parser.add_argument('--test_path', type=str, default='data/sample_test.jsonl')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--output_path', type=str, default='outputfile.txt')
        parser.add_argument('--model_path', type=str, default='models/cs1190444_cs1190673_model')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=50)
        parser.add_argument('--model', type=str, default='gpt2')
        parser.add_argument('--use_random_split', type=bool, default=False)
        parser.add_argument('--split_ratio', type=float, default=0.8)
        parser.add_argument('--init_lr', type=float, default=1e-5)
        parser.add_argument('--warmup_steps', type=int, default=1000)
        parser.add_argument('--optimizer', type=str, default='adamw')
        parser.add_argument('--save_wandb', type=bool, default=False)
        parser.add_argument('--scheduler', type=str, default='linear_warmup')
        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()
    

def parse_input(data, tokenizer, model):
    outputs = []
    for d in data:
        input_str = d['input']
        history_str = ' '.join([x['user_query'] + ' ' + x['response_text'] for x in d['history']])
        user_lists_str = ' '.join([l['name'] + ' ' + ' '.join(l['items']) for l in d['user_lists']])
        user_notes_str = ' '.join([n['name'] + ' ' + n['content'] for n in d['user_notes']])
        user_contacts_str = ' '.join(d['user_contacts'])

        text = input_str + ' ' + history_str + ' ' + user_lists_str + ' ' + user_notes_str + ' ' + user_contacts_str

        input_ids = tokenizer.encode(text, return_tensors='pt')
        outputs.append(model.generate(input_ids, max_length=512, do_sample=True))
        
    result = []
    for i in range(len(outputs)):
        output_str = tokenizer.decode(outputs[i][0], skip_special_tokens=True)
        output_str = output_str.replace('(', '( ')
        output_str = output_str.replace(')', ' )')
        result.append({'output': output_str})
    return result


def text_intent_slot_values(s:str):
    intent = s.rstrip().lstrip().split(' ')[0]
    slot_values = s.rstrip().lstrip().split(' ')[1:]
    return intent, slot_values


def get_intent_count(data: DataLoader):
    intent_counter = defaultdict(int)
    for i in range(len(data)):
        intent = text_intent_slot_values(data[i])[0]
        intent_counter[intent] += 1
    print("Number of different intents found =",len(intent_counter))
    print("Number of data points =",sum(intent_counter.values()))
    # print key, value in sorted order
    for key, value in sorted(intent_counter.items(), key=lambda item: item[1], reverse=True):
        print(key,value)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)

def create_training_args(args):
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=8,   # batch size per device during training
        per_device_eval_batch_size=8,    # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=1000,              # log every x steps
        evaluation_strategy='steps',     # evaluation strategy to adopt during training
        save_total_limit=1,              # number of total checkpoints to save
        save_steps=5000,                 # save checkpoint every x steps
        learning_rate=5e-5,
    )
    return training_args