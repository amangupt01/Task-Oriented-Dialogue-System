
from imports import *

from dataloader import TODDataset
from utils import *
from evaluations import Evaluator
from gpt2 import GPT2_TOD
# from t5 import T5_TOD

if __name__ == '__main__':
    seed_everything(42)
    parser = ArgumentParser().parse_args()
    args = vars(parser)
    print(args)

    if (args['mode'] == 'train'):

        if (args['model'].startswith('gpt2')):
            model = GPT2_TOD(args)

        model.train(args)

    if (args['mode'] == 'test'):
        pass
        