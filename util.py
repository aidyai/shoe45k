from typing import Dict
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ruamel.yaml as yaml
import numpy as np
import evaluate


def load_config(config_path: str) -> Dict:
    yaml_loader = yaml.YAML(typ='rt')
    with open(config_path, 'r') as file:
        return yaml_loader.load(file)


def collate_fn_cls(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples]) 
    return {"pixel_values": pixel_values, "labels": labels}



def compute_metrics_cls(eval_pred):
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_accuracy = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision = metric_precision.compute(predictions=predictions, references=labels, average='macro')["precision"]
    recall = metric_recall.compute(predictions=predictions, references=labels, average='macro')["recall"]
    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "accuracy": accuracy}