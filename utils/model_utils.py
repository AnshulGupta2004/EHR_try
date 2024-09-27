# utils/model_utils.py

import random
import numpy as np
import torch

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load(model, load_model_path, args):
    # Example loading function if needed
    checkpoint = torch.load(load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = None
    scheduler = None
    step = checkpoint.get('step', 0)
    best_metric = checkpoint.get('best_metric', None)
    return model, optimizer, scheduler, args, step, best_metric
