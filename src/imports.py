import jsonlines, argparse, random, os, sys, math, pickle
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
from transformers import TrainingArguments, set_seed, AdamW, get_linear_schedule_with_warmup

# def get_least_utilized_gpu():
#     gpu_memory = []
#     for i in range(torch.cuda.device_count()):
#         gpu_memory.append(torch.cuda.memory_allocated(i))
#     device_idx = gpu_memory.index(min(gpu_memory))
#     return torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

# device = get_least_utilized_gpu()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")