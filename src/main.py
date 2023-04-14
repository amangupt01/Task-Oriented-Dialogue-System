
from imports import *

from dataloader import DataLoader
from utils import ArgumentParser, get_intent_count, get_slot_key_count

if __name__ == '__main__':
    parser = ArgumentParser().parse_args()
    args = vars(parser)
    # print(args)
    # print(args['train_path'])
    train_data = DataLoader(args['train_path'])
    dev_data = DataLoader(args['dev_path'])
    get_slot_key_count(train_data)


    pass