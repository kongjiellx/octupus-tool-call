import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import json
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithFlattening
from transformers.trainer_pt_utils import LabelSmoother
from accelerate import PartialState


IGNORE_INDEX = LabelSmoother.ignore_index
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str]
    tokenizer_name_or_path: Optional[str]
    model_max_length: int = field(default=4096)


@dataclass
class DataArguments:
    data_path: str
    dataset_cache_dir: str = field()
    num_proc: int = field(default=10)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def load_model(model_path):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.train()
    return model


def buid_instruction_dataset(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    preprocessing_num_workers=None,
    dataset_cache_dir=None,
    seed=42,
):
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."
    FUNCTION_SUFIX = "你可以使用这些工具来帮助用户解决问题: "
    IGNORE_INDEX = -100

    def tokenization(example):
        conversation, tools = example['conversation'], example.get('tools')

        token_ids = []
        labels = []

        if conversation[0]["role"] == "system":
            system_msg = f"<|SYSTEM|>{conversation[0]['content']}"
            conversation = conversation[1:]
        else:
            system_msg = f"<|SYSTEM|>{DEFAULT_SYSTEM_MESSAGE}"
        if tools:
            system_msg += f"\n\n{FUNCTION_SUFIX}{tools}"
        system_ids = tokenizer.encode(system_msg)
        token_ids += system_ids
        labels += [IGNORE_INDEX] * len(system_ids)

        for i, msg in enumerate(conversation):
            if msg["role"] == "user":
                user_ids = tokenizer.encode("<|USER|>" + msg['content'])
                token_ids += user_ids
                labels += [IGNORE_INDEX] * len(user_ids)
            elif msg["role"] == "assistant":
                assistant_text = "<|ASSISTANT|>" + (f"<|CONTENT|>{msg['content']}" if msg["content"] else "")
                if fcs := msg.get("tool_calls"):
                    for fc in fcs:
                        assistant_text += f"<|FUNCTION_CALL|>{fc['function']['name']}<|PARAMETERS|>{fc['function']['arguments']}"
                assistant_ids = tokenizer.encode(assistant_text) + [tokenizer.eos_token_id]
                token_ids += assistant_ids
                labels += [IGNORE_INDEX] * 1 + assistant_ids[1:]
            elif msg["role"] == "tool":
                function_ids = tokenizer.encode("<|FUNCTION_OUTPUT|>" + msg['content'])
                token_ids += function_ids
                labels += [IGNORE_INDEX] * len(function_ids)
        results = {
            'input_ids': token_ids,
            'labels': labels,
        }
        return results

    def pad(example):
        return {
            "input_ids": example["input_ids"] + [tokenizer.pad_token_id] * (max_seq_length - len(example["input_ids"])),
            "labels": example["labels"] + [-100] * (max_seq_length - len(example["input_ids"]))
        }

    dataset_and_split = []
    with open(data_path, "r") as fp:
        for line in fp:
            if line.strip():
                dataset_and_split.append(line.strip().split(":"))
    rank0_print(f"train datasets: {dataset_and_split}")

    all_datasets = []
    for ds, split, ratio in dataset_and_split:
        ratio = float(ratio)
        rank0_print(ds, split, ratio)
        raw_dataset = load_dataset(ds, split=split, cache_dir=dataset_cache_dir)
        if ratio > 1 + 1e-8:
            copy_dataset = concatenate_datasets([raw_dataset] * int(ratio)) if int(ratio) > 1 else raw_dataset
            select_dataset = raw_dataset.shuffle(seed=seed).select(range(0, int(len(raw_dataset) * (ratio - int(ratio)))))
            raw_dataset = concatenate_datasets([copy_dataset, select_dataset])
        elif ratio < (1 - 1e-8):
            raw_dataset = raw_dataset.shuffle(seed=seed).select(range(0, int(len(raw_dataset) * ratio)))
        tokenized_dataset = raw_dataset.map(
            tokenization,
            num_proc=preprocessing_num_workers,
            remove_columns=set(raw_dataset.features.keys()) - {'input_ids', 'labels'},
        ).filter(lambda x: len(x["input_ids"]) <= max_seq_length, num_proc=preprocessing_num_workers)
        all_datasets.append(tokenized_dataset)
        rank0_print(tokenized_dataset)
    all_datasets = concatenate_datasets(all_datasets).shuffle(seed=seed)
    return dict(train_dataset=all_datasets, eval_dataset=None)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with PartialState().local_main_process_first():
        data_module = buid_instruction_dataset(
            data_args.data_path,
            tokenizer,
            max_seq_length=model_args.model_max_length,
            preprocessing_num_workers=data_args.num_proc,
            dataset_cache_dir=data_args.dataset_cache_dir,
        )

    model = load_model(model_args.model_name_or_path)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithFlattening(),
        args=training_args,
        **data_module,
    )

    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )


if __name__ == "__main__":
    train()
