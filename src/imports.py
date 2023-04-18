import jsonlines, argparse, random, os, sys
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
from transformers import TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, set_seed
from transformers import AdamW, get_linear_schedule_with_warmup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")