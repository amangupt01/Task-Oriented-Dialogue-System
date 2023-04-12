from imports import *

from dataloader import DataLoader
from utils import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser().parse_args()
    args = vars(parser)

    train_data = DataLoader(args['train_path'])
    dev_data = DataLoader(args['dev_path'])
    print(args)
    pass