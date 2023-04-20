
from imports import *

from utils import *
from evaluations import Evaluator
from gpt2 import GPT2_TOD
from t5 import T5_TOD

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
            pickle.dump(model, fp)

    if (args['mode'] == 'test'):


        try:
            with open(args['model_path'], 'rb') as fp:
                model = pickle.load(fp)
        except:
            print("Model loading failed")
            sys.exit(0)

        model.predict(args)
        