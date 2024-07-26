import pandas as pd
import ruamel.yaml as yaml
from typing import Dict
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForImageClassification, 
    AutoImageProcessor,
    AutoProcessor, 
    BlipForConditionalGeneration,
    AdamW
)
from peft import LoraConfig, get_peft_model
from data.dataset import Shoe45kDataset, BlipDataset
from util import load_config, collate_fn_cls, blip_collate_fn, compute_metrics_cls
from datasets import load_dataset

def train(config: Dict):
    task = config["task"]
    model_checkpoint = config["model_checkpoint"]
    wandb_name = config["wandb_name"]
    num_classes = config["num_classes"]
    lora_r = config["lora_r"]
    lora_alpha = config["lora_alpha"]
    lora_target_modules = config["lora_target_modules"]
    lora_dropout = config["lora_dropout"]
    lora_bias = config["lora_bias"]


    if task == "classification":

        label_mapping = {
          "Sneakers": 0,
          "Boot": 1,
          "Sandals": 2,
          'Crocs': 3, 
          'Heels': 4, 
          'Dressing Shoe': 5,
          }

        # Load the dataset
        train = load_dataset(config["dataset_name"], split='train')
        val = load_dataset(config["dataset_name"], split='validation')

        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=model_checkpoint,
            num_labels=num_classes,
        )
        #cls_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

        # Create the custom dataset
        train_dataset_cls = Shoe45kDataset(train, phase='train', label_mapping=label_mapping)
        val_dataset_cls = Shoe45kDataset(val, phase='val', label_mapping=label_mapping)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=["classifier"],
        )
        compute_metrics = compute_metrics_cls
        data_collator_cls = collate_fn_cls

    elif task == "captioning":

        model = BlipForConditionalGeneration.from_pretrained(model_checkpoint)
        processor = AutoProcessor.from_pretrained(model_checkpoint)

        # Load the dataset
        hf_dataset = load_dataset(config["dataset_name"], split='train')

        # Create the custom dataset
        train_dataset_blip = BlipDataset(hf_dataset, processor)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
        )
        
        data_collator = blip_collate_fn



    lora_model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=config["training_args"]["output_dir"],
        remove_unused_columns=config["training_args"]["remove_unused_columns"],
        evaluation_strategy=config["training_args"]["evaluation_strategy"],
        save_strategy=config["training_args"]["save_strategy"],
        learning_rate=config["training_args"]["learning_rate"],
        per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training_args"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
        fp16=config["training_args"]["fp16"],
        num_train_epochs=config["training_args"]["num_train_epochs"],
        logging_steps=config["training_args"]["logging_steps"],
        load_best_model_at_end=config["training_args"]["load_best_model_at_end"],
        metric_for_best_model=config["training_args"]["metric_for_best_model"],
        push_to_hub=config["training_args"]["push_to_hub"],
        label_names=config["training_args"]["label_names"],
        report_to=config["training_args"]["report_to"],
        run_name=wandb_name,
    )


    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset_cls if task == "classification" else train_dataset_blip,
        eval_dataset=val_dataset_cls if task == "classification" else None,
        tokenizer=None if task == "classification" else processor,
        compute_metrics=compute_metrics if task == "classification" else None,
        data_collator=data_collator_cls if task == "classification" else data_collator,
        )


    train_results = trainer.train()
    return train_results


if __name__ == "__main__":
    config_path = "/content/shoe45k/configs/vit.yaml"  # Change to your desired config file
    config = load_config(config_path)
    train(config)