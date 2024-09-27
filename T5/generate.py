# T5/generate.py

import time
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import json
import os

def generate_sql(pipe, eval_dataset, args, collator, verbose=0):
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
    
    output_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, args.output_file)
    
    out_eval = {}
    
    with open(output_file, 'w') as f:
        f.write('{\n')  # Start of JSON object
        first_entry = True

        for idx, batch in enumerate(tqdm(dataloader, desc="Generating SQL"), 1):
            input_texts = batch['inputs']
            db_ids = batch['db_id']
            is_impossibles = batch['is_impossible']
            data_ids = batch['id']

            # Generate predictions using the pipeline
            outputs = pipe(
                input_texts,
                max_new_tokens=args.max_length,
                num_beams=args.num_beams,
                do_sample=(args.num_beams == 1),
                num_return_sequences=args.num_samples,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping
            )
            
            # If num_return_sequences > 1, adjust indexing
            for i, output in enumerate(outputs):
                # If multiple sequences per input, map accordingly
                pred = output['generated_text'].strip()
                real = batch['labels'][i].strip()
                result = {
                    'question': input_texts[i],
                    'real': real,
                    'pred': pred,
                    'db_id': db_ids[i],
                    'is_impossible': is_impossibles[i],
                    'sequence_entropy': ()  # Entropy calculation is not implemented
                }
                out_eval[data_ids[i]] = result

                # Write to file incrementally
                if not first_entry:
                    f.write(',\n')
                else:
                    first_entry = False
                json.dump(data_ids[i], f)
                f.write(': ')
                json.dump(result, f)

                if verbose > 0 and idx % 10 == 0:
                    print(f'Processed {idx} samples', end='\r')

        f.write('\n}')  # End of JSON object

    if verbose > 0:
        print(f"Inference took {round(time.time() - start_time, 6)} secs")

    return out_eval
