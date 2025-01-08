import yaml
import os
import json
import argparse
from typing import Optional, Literal, List
import copy
from functools import partial
import re

from datasets import Dataset
from tqdm.auto import tqdm
from unsloth import FastLanguageModel
from pydantic import BaseModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


def extract_llama_answer(output):
    # Use regex to find the assistant's response
    match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>', output, re.DOTALL)

    if match:
        assistant_answer = match.group(1)
        return assistant_answer
    else:
        return None


class Config(BaseModel):
    max_seq_length: int = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype: Optional[str] = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool = True # Use 4bit quantization to reduce memory usage. Can be False.
    fourbit_model: Literal[
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
        "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
        "unsloth/Phi-3-medium-4k-instruct",
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
        "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct"
    ] = "unsloth/Llama-3.2-1B-bnb-4bit"
    token: str
    r: int = 16
    use_rslora: bool = False
    training_args: dict
    run_name: str
    system_path: str
    conversations_path: str


def get_output_dir(config: Config) -> str:
    path = os.path.join("results", "llm", config.run_name)
    os.makedirs(path, exist_ok=True)
    return path

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }


def convert_to_conversations(example, system_path: str):
    system = open(system_path, "r").read()
    conversation = example.get("text", "")
    return {
        "conversations": [
            {"from": "system", "value": system},
        ] + conversation["conversations"],
    }


def main(
    config_path: str
):
    config = Config.model_validate(yaml.safe_load(open(config_path, "r")))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.fourbit_model, 
        max_seq_length = config.max_seq_length,
        dtype = config.dtype,
        load_in_4bit = config.load_in_4bit,
        token=config.token
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = config.r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = config.use_rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
    formatting_fn = partial(formatting_prompts_func, tokenizer=tokenizer)

    train_conversations = [
        json.loads(line) for line in open(config.conversations_path, "r").readlines()
    ]
    train_dataset = Dataset.from_dict({"text": train_conversations})
    train_dataset = train_dataset.map(convert_to_conversations, fn_kwargs={"system_path": config.system_path}, remove_columns=train_dataset.column_names)

    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_fn, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = config.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            optim = "adamw_8bit",
            output_dir = get_output_dir(config) + "/checkpoints",
            report_to = "none", # Use this for WandB etc
            save_strategy = "epoch",
            **config.training_args
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer.train()

    output_dir = get_output_dir(config)
    model_path = os.path.join(output_dir, "best_checkpoint")
    config_copy_path = os.path.join(output_dir, "config_copy.yaml")
    
    model.save_pretrained(model_path + "/model")
    tokenizer.save_pretrained(model_path + "/tokenizer")
    
    with open(config_copy_path, "w") as f:
        yaml.dump(config.model_dump(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
