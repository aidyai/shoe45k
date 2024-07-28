import torch
from torch.optim import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning.core import LightningModule



class BlipTransformerModule(LightningModule, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        processor = AutoProcessor.from_pretrained(config["processor_checkpoint"])
        model = Blip2ForConditionalGeneration.from_pretrained(
            config["model_checkpoint"],
            device_map="auto", 
            load_in_4bit=True
        )

        lora_config = config["lora_config"]
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            target_modules=lora_config["target_modules"]
        )
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        self.processor = processor
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
    
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs["loss"]
        if torch.isnan(loss):
            print("NaN loss detected during training")
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
          )
        loss = outputs["loss"]
        if torch.isnan(loss):
            print("NaN loss detected during validation")
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return AdamW(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )