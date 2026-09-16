import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import Dataset

df = pd.read_csv("dataset/labels.csv")

model_id = "datalab-to/surya-ocr-2"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

class OCRDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            image_path = os.path.join("dataset", image_path)
            
        return {
            "image": Image.open(image_path).convert("RGB"),
            "text": row["text"]
        }

def custom_collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs["labels"] = inputs["input_ids"].clone()
    
    inputs["labels"][inputs["labels"] == processor.tokenizer.pad_token_id] = -100
    
    return inputs

dataset = OCRDataset(df)

training_args = TrainingArguments(
    output_dir="./custom_surya_model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    save_steps=10,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=custom_collate_fn
)

trainer.train()
model.save_pretrained("./custom_surya_model")
processor.save_pretrained("./custom_surya_model")