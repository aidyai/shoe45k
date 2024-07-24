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


def compute_metrics_cls(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
