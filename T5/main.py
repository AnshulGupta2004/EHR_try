# T5/main.py

import sys
import os
# Add the project root directory to sys.path
# This assumes that 'main.py' is inside 'EHRSQL/T5/'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import json
import numpy as np
import argparse
import yaml

import torch
from utils.model_utils import set_seeds
from utils.optim import set_optim
from utils.dataset import EHRSQL_Dataset, DataCollator
from utils.logger import init_logger
from config import Config

from transformers import pipeline  # Updated import for pipeline
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login

login(token = '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')

if __name__ == '__main__':
    args = Config()
    args.get_param(use_model_param=True,
                   use_eval_param=True,
                   use_optim_param=True)
    args.parser.add_argument('--config', required=True, type=str)
    args.parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str)
    args = args.parse()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    for k, v in config.items():
        if config[k]:
            setattr(args, k, config.get(k))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES if args.device == 'cuda' else ""
    
    if torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device(f'cuda:{args.CUDA_VISIBLE_DEVICES}')
        print(f'Current device: cuda:{args.CUDA_VISIBLE_DEVICES}')
    else:
        device = torch.device('cpu')
        print('Current device: cpu')
    
    set_seeds(args.random_seed)
    
    output_path = os.path.join(args.output_dir, args.exp_name)
    if os.path.exists(output_path):
        raise Exception(f"Directory already exists ({output_path})")
    
    logger = None
    if args.mode == 'train':
        logger = init_logger(output_path, args)
        logger.info(args)


    # Initialize Hugging Face pipeline for text generation
    model_id = args.model_name  # Ensure 'model_name' is specified in your config.yaml
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16, #if args.device == 'cuda' else torch.float32,
        device_map="auto",
    )

    
    data_collator = DataCollator(tokenizer=None, return_tensors='pt')  # Tokenizer handled by pipeline

    if args.mode == 'train':
        train_dataset = EHRSQL_Dataset(
            path=args.train_data_path, 
            tokenizer=None,  # Tokenizer handled by pipeline
            args=args, 
            data_ratio=args.training_data_ratio
        )
        valid_dataset = EHRSQL_Dataset(
            path=args.valid_data_path, 
            tokenizer=None,  # Tokenizer handled by pipeline
            args=args
        )
        if logger:
            logger.info(f"Loaded {len(train_dataset)} training examples from {args.train_data_path}")
            logger.info(f"Loaded {len(valid_dataset)} valid examples from {args.valid_data_path}")
    elif args.mode == 'eval':
        test_dataset = EHRSQL_Dataset(
            path=args.test_data_path, 
            tokenizer=None,  # Tokenizer handled by pipeline
            args=args, 
            include_impossible=True
        )
        print(f"Loaded {len(test_dataset)} test examples from {args.test_data_path}")

    if args.load_model_path and args.mode != 'train':
        # Optionally load a fine-tuned model from a directory
        pipe = pipeline(
            "text-generation",
            model=args.load_model_path,
            torch_dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32,
            device_map="auto" if args.device == 'cuda' else None,
        )
        if logger:
            logger.info(f"Loaded model from {args.load_model_path}")
        else:
            print(f"Loaded model from {args.load_model_path}")

    if torch.cuda.device_count() > 1 and args.mode == 'train':
        pipe.model = torch.nn.DataParallel(pipe.model)

    if args.mode == 'eval':
        from T5.generate import generate_sql
        print("Start inference")
        out_eval = generate_sql(pipe=pipe, eval_dataset=test_dataset, args=args, collator=data_collator, verbose=1)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, args.output_file), 'w') as f:
            json.dump(out_eval, f)
    else:
        from trainer_t5 import train
        print("Start training")
        optimizer, scheduler = set_optim(args, None)  # Pass None as model is handled by pipeline
        step = 0
        if args.eval_metric == 'loss':
            best_metric = np.inf
        elif args.eval_metric == 'esm':
            best_metric = -np.inf
        train(
            pipe=pipe,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            args=args,
            collator=data_collator,
            best_metric=best_metric,
            checkpoint_path=output_path,
            logger=logger
        )
