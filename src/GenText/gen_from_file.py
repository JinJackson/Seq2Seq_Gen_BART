

import argparse
import numpy as np
import torch
import glob
import logging
import os
import random
import shutil


from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer,
                                  BartConfig, BartTokenizer, BartForConditionalGeneration)
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, DistributedSampler
import sys
sys.path.append('../')
from utils import TrainDataset, GenDataset


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--language", default=None, type=str, required=True)
    parser.add_argument("--gen_type", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_name", default='', type=str, 
                        help="The training file name.")

    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_src_len", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--max_tgt_len", default=512, type=int,
                        help="Optional target sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--load_output_model", action='store_true',
                        help="Whether to load model from output directory.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    return args


def get_source_data(source_file):
    all_data = []
    with open(source_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            # print(line)
            all_data.append(line.strip())
    assert len(all_data) != 0
    return all_data


def load_and_cache_examples(tokenizer):
    dataset = GenDataset(tokenizer, data_dir=args.data_dir, max_source_length=args.max_src_len)
    return dataset
    

if __name__ == "__main__":
    args = get_args()
    batch_size = args.per_gpu_eval_batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    source_file = args.data_dir
    all_data = get_source_data(source_file)

    MODEL_CLASSES = {
        'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
        'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    }


        

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path).to(device)

    if args.language == 'cn':
        tokenizer_class = BertTokenizer
    elif args.language == 'en':
        tokenizer_class = BartTokenizer
    
    print("tokenizer_class:", tokenizer_class)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    eval_dataset = load_and_cache_examples(tokenizer)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=eval_dataset.collate_fn)

    # Gen!!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", batch_size)


    preds = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            with torch.no_grad():
                generated_ids = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=True, num_beams=5, 
                    length_penalty=0.6, max_length=32, repetition_penalty=2.0, decoder_start_token_id=101)
                gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                gen_text = [t if t.endswith('？') else t+' ？' for t in gen_text]
                preds += gen_text
    # written_file = args.data_dir + '_generated'
    original_data = []
    with open(args.data_dir, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        original_data = [line.strip() for line in lines]
    
    if args.gen_type not in ['para', 'nonpara']:
        print('gen_type wrong, plz input \"para\" or \"nonpara\"')
    
    label = 1 if args.gen_type == 'para' else 0
    print(len(original_data), len(preds))
    assert len(original_data) == len(preds)

    written_file = args.output_dir
    with open(written_file, "w", encoding='utf-8') as writer:
        for origin, gen in zip(original_data, preds):
            gen = ''.join(gen.split())
            if origin.rstrip('？').rstrip('?').rstrip('。').rstrip('！').rstrip('.') == gen.rstrip('？').rstrip('?').rstrip('。').rstrip('！').rstrip('.'):
                continue

            writer.write(origin + '\t' + gen + '\t' + str(label) + '\n')
            