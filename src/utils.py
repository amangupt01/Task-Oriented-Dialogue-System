from imports import *

class ArgumentParser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', type=str, default='data/train.jsonl')
        parser.add_argument('--dev_path', type=str, default='data/dev.jsonl')
        parser.add_argument('--test_path', type=str, default='data/sample_test.jsonl')
        parser.add_argument('--output_path', type=str, default='outputfile.txt')
        parser.add_argument('--model_path', type=str, default='cs1190444_cs1190673_model')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_epochs', type=int, default=10)

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