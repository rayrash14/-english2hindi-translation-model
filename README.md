# English-to-Hindi Translation Model

This repository contains an English-to-Hindi translation model fine-tuned using the Helsinki-NLP/opus-mt-en-hi model. The model has been trained and evaluated on the IITB English-Hindi dataset and deployed on the Hugging Face Hub.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing the Model](#testing-the-model)
- [Saving and Uploading](#saving-and-uploading)
- [Loading the Model from Hugging Face](#loading-the-model-from-hugging-face)

## Introduction

This project aims to develop a robust English-to-Hindi translation model using the `Helsinki-NLP/opus-mt-en-hi` model from Hugging Face. The model is fine-tuned on the IITB English-Hindi dataset to improve translation quality.

## Installation

To set up the environment, install the necessary dependencies:

```bash
pip install datasets transformers sacrebleu torch sentencepiece transformers[sentencepiece] evaluate huggingface_hub
```

## Dataset

We use the IITB English-Hindi dataset, which can be loaded as follows:

```python
from datasets import load_dataset
raw_datasets = load_dataset("cfilt/iitb-english-hindi")
```

A subset of 50,000 examples is used for training and validation.

```python
small_dataset = raw_datasets["train"].select(range(50000))
```

## Training

We split the dataset into training (90%) and validation (10%) sets:

```python
train_size = int(0.9 * len(small_dataset))
train_dataset = small_dataset.select(range(train_size))
val_dataset = small_dataset.select(range(train_size, len(small_dataset)))
```

The dataset is tokenized using the `AutoTokenizer` from Hugging Face:

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Data preprocessing is performed to tokenize both input and target texts:

```python
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["hi"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

The tokenized datasets are created:

```python
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=4)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, num_proc=4)
```

The model is fine-tuned using `Seq2SeqTrainer`:

```python
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    "opus-mt-en-hi-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

## Evaluation

The model is evaluated using BLEU score with `sacrebleu`:

```python
import evaluate
metric = evaluate.load("sacrebleu")
```

## Testing the Model

To test a sample translation:

```python
sample_text = "I feel good"
inputs = tokenizer(sample_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)
translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("English:", sample_text)
print("Hindi Translation:", translated_text[0])
```

## Saving and Uploading

The trained model is saved locally:

```python
save_path = "./hindi_translation_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

The model is then uploaded to the Hugging Face Hub:

```python
from huggingface_hub import HfApi
api = HfApi()
repo_id = "your_username/english2hindi-translation-model"
api.create_repo(repo_id, exist_ok=True)

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

## Loading the Model from Hugging Face

To reload the model:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
repo_id = "your_username/english2hindi-translation-model"
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

## Testing Translation from Hugging Face Hub

```python
sample_text = "Hello, how are you?"
inputs = tokenizer(sample_text, return_tensors="pt")
outputs = model.generate(**inputs)
translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print("Translated Output:", translated_text[0])
```

## Conclusion

This repository provides an end-to-end pipeline for training, evaluating, and deploying an English-to-Hindi translation model. The model is fine-tuned using the IITB dataset and can be accessed through the Hugging Face Hub for real-time translation.

