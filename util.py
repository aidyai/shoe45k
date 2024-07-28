from typing import Dict
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ruamel.yaml as yaml



def load_config(config_path: str) -> Dict:
    yaml_loader = yaml.YAML(typ='rt')
    with open(config_path, 'r') as file:
        return yaml_loader.load(file)


def collate_fn_cls(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}



def blip_collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


def compute_metrics_cls(eval_pred):

    
    metric1 = load("precision")
    metric2 = load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    predictions = predictions.flatten() if len(predictions.shape) > 1 else predictions
    labels = labels.flatten() if len(labels.shape) > 1 else labels

    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    return {"precision": precision, "recall": recall}