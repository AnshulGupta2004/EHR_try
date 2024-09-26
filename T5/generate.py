# T5/generate.py

import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from utils.groq_api import GroqAPI  # Import the GroqAPI class
from tqdm import tqdm  # Ensure tqdm is imported for progress bars

def generate_sql(model, eval_dataset, args, collator, verbose=0):
    file_name = args.config.split('/')[-1]
    start_time = time.time()
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.eval_batch_size, 
        drop_last=False,
        collate_fn=collator
    )
    tokenizer = eval_dataset.tokenizer

    groq_api = GroqAPI()  # Initialize GroqAPI

    out_eval = {}
    for idx, batch in enumerate(tqdm(dataloader, desc="Generating SQL"), 1):
        input_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['inputs']]
        db_ids = batch['db_id']
        is_impossibles = batch['is_impossible']
        data_ids = batch['id']

        for i, input_text in enumerate(input_texts):
            # If schema info needs to be included, adjust the prompt accordingly
            prompt = input_text  # Modify if necessary to include schema
            pred = groq_api.generate_sql(prompt=prompt)

            # Assuming the real query is available in batch['labels']
            real = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)

            result = {
                'question': input_text,
                'real': real,
                'pred': pred,
                'db_id': db_ids[i],
                'is_impossible': is_impossibles[i],
                'sequence_entropy': ()  # Entropy not available via API
            }

            out_eval[data_ids[i]] = result

            if verbose > 0 and idx % 10 == 0:
                print(f'Processed {idx} samples', end='\r')

    if verbose > 0:
        print(f"Inference took {round(time.time() - start_time, 6)} secs")

    return out_eval
