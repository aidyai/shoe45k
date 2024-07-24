import ruamel.yaml as yaml
from typing import List, Dict
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoModelForImageClassification, 
    AutoProcessor,
    AutoModelForSeq2SeqLM
)
from peft import LoraConfig, get_peft_model
from dataset import load_dataset


def train(
    config: Dict,
):
    task = config["task"]
    pretrained_model = config["pretrained_model"]
    wandb_name = config["wandb_name"]
    num_classes = config.get("num_classes", 6)
    lora_r = config["lora_r"]
    lora_alpha = config["lora_alpha"]
    lora_target_modules = config["lora_target_modules"]
    lora_dropout = config["lora_dropout"]
    lora_bias = config["lora_bias"]


    config_path = './config/pretrain.yaml'
    yaml_loader = yaml.YAML(typ='rt')
    config = yaml_loader.load(open(config_path, 'r'))

    train_ds, val_ds = load_dataset(task)





    if task == "classification":
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=num_classes,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            modules_to_save=["classifier"],
        )

    elif task == "captioning":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model, 
            load_in_8bit=True
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
        )

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
        run_name=config["training_args"]["run_name"],
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None if task == "classification" else AutoProcessor.from_pretrained(pretrained_model),
        compute_metrics=None,  # Define your compute_metrics function as needed
        data_collator=None  # Define your collate_fn function as needed
    )

    train_results = trainer.train()
    return train_results

if __name__ == "__main__":
    config_path = "config_classification.yml"  # Change to your desired config file
    config = load_config(config_path)

    train(config)
