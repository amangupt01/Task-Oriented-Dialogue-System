from imports import *

class DataLoader():
    def __init__(self, data_path):
        data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                data.append(self.get_input(obj))
        self.data = data
        print(len(data))
        print(data[0])
        print(data[1])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_input(obj):
        input_str = obj['input']
        history_str = ' '.join([x['user_query'] + ' ' + x['response_text'] for x in obj['history']])
        user_lists_str = ' '.join([l['name'] + ' ' + ' '.join(l['items']) for l in obj['user_lists']])
        user_notes_str = ' '.join([n['name'] + ' ' + n['content'] for n in obj['user_notes']])
        user_contacts_str = ' '.join(obj['user_contacts'])
        return input_str + ' ' + history_str + ' ' + user_lists_str + ' ' + user_notes_str + ' ' + user_contacts_str
