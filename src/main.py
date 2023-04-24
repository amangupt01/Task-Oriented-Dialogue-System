
from imports import *

from utils import *
from evaluations import Evaluator
from gpt2 import GPT2_TOD
from t5 import T5_TOD, T5_Dataset

if __name__ == '__main__':
    seed_everything(42)
    parser = ArgumentParser().parse_args()
    args = vars(parser)
    print(args)

    if (args['mode'] == 'train'):

        if (args['model'].startswith('gpt2')):
            model = GPT2_TOD(args)
        elif (args['model'].startswith('t5')):
            model = T5_TOD(args)
        model.train(args)

        with open(args['model_path'], 'wb') as fp:
            pickle.dump({
                "model_name": args['model'],
                "model" : model.model,
                "tokenizer": model.tokenizer
            }, fp)

    if (args['mode'] == 'test'):

        if (args['model'].startswith('t5')):
            try:
                with open(args['model_path'], 'rb') as fp:
                    model_dict = pickle.load(fp)
                    model = model_dict['model']
                    tokenizer = model_dict['tokenizer']
            except:
                print("Model loading failed")
                sys.exit(0)
            
            print("Model loaded!")
            test_data = T5_Dataset([args['test_path']], tokenizer, val=True, aux_input=args['aux_input'])
            test_dataloader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
            outfile = open(args['output_path'], 'w')
            predictions = []
            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['input_attention_mask'].to(device)
                    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, do_sample=True, top_p=0.95, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
                    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    predictions.extend(outputs)
                outfile.write('\n'.join(predictions))
        