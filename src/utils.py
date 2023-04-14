from imports import *
from dataloader import DataLoader

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

def recur_get_slot_key(l: list, slot_counter: dict):
    if len(l) == 0: return
    if '«' not in l and '(' not in l:
        return
    first_index = len(l)-1
    if '«' in l:
        first_index = min(first_index, l.index('«'))
    if '(' in l:
        first_index = min(first_index, l.index('('))
    assert first_index != len(l)-1, "Error in parsing-1"
    # slot_counter[" ".join(l[:first_index])] += 1
    for slots in l[:first_index]:
        slot_counter[slots] += 1

    char_at_first_index = l[first_index]
    if char_at_first_index == '«':
        second_index = l.index('»')
        recur_get_slot_key(l[second_index+1:], slot_counter)
    else:
        net_bracket = 1
        found_end_pos = -1
        for i in range(first_index+1, len(l)):
            if l[i] in ['«', '(']:
                net_bracket += 1
            elif l[i] in ['»', ')']:
                net_bracket -= 1
            if net_bracket == 0:
                found_end_pos = i
                break
        assert found_end_pos != -1, "Error in parsing-2"
        recur_get_slot_key(l[first_index+1:found_end_pos], slot_counter)
        recur_get_slot_key(l[found_end_pos+1:], slot_counter)
    """
         slot << value >>     ->.  print slot
         slot ( recursion ).  -> print slot + recurse
         ['a', '(', "b", "(", "c", '«', 'Rachel', '»', ")", ")", "aman", '«', 'nan mam', '»']
    """

def get_slot_key_count(data: DataLoader):
    slot_counter = defaultdict(int)
    for i in range(len(data)):
        slot_values = text_intent_slot_values(data[i])[1]
        relevant = slot_values[1:-1]
        print(relevant)
        break
        # relevant =  ['a', '(', "b", "(", "c", '«', 'Rachel', '»', ")", ")", "aman", '«', 'nan mam', '»']
        recur_get_slot_key(relevant, slot_counter)
        # for x in relevant:
        #     flag = False
        #     if (x == '»'):
        #         while(stack[-1] != '«'):
        #             stack.pop()
        #         stack.pop()
        #         c -= 1
        #     elif (x == ')'):
        #         while(stack[-1] != '('):
        #             stack.pop()
        #         stack.pop()
        #         c -= 1
        #     else:
        #         if x == '«' or x == "(": 
        #             c += 1
        #         else:
        #             flag = True
        #         stack.append(x)
        #     if (c == 0 and not flag):
        #         if len(stack) != 0:
        #             print(" ".join(stack))
        #         stack = []
        # break
    print("Number of different slots found =",len(slot_counter))
    
    # print key, value in sorted order
    for key, value in sorted(slot_counter.items(), key=lambda item: item[1], reverse=True):
        print(key,value)
