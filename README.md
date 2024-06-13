# climate_attitude_LM

## Overview
https://huggingface.co/Kevintu/climate-attitude-LM


This language model is designed to assess the attitude expressed in texts about **climate change**. It categorizes the attitude into three types: risk, neutral, and opportunity. These categories correspond to the negative, neutral, and positive classifications commonly used in sentiment analysis. We employed a fine-tuning approach to adapt the final layer of the "cardiffnlp/twitter-roberta-base-sentiment-latest" model using a training dataset from "climatebert/climate_sentiment."

Our model demonstrates exceptional performance. When evaluated using the test dataset from "climatebert/climate_sentiment," it achieves an accuracy of 89% and an F1 score of 89%.

**Note** that you should paste or type a text concerning the **climate change** in the API input bar or using the testing code. Otherwise, the model does not work. e,.g, the example input, "The Deputy Mayor also said that while RRF regulations are clear at EU level, bureaucracy at national level in Italy has led to confusion for cities. “At national level, the implementation of RRF funds is not straightforward, there are ministries that have their own rules and cities get different information from different people,” she explained.
Adding to the discussion on whether a simpler, faster model, led by national governments, is best for dispersing EU funds, Boni stressed the importance of finding the right balance.“

Please cite: ``Sun., K, and Wang, R. 2024. The fine-tuned language model for detecting human attitudes to climate changes'' if you use this model.


   
# Training implementation

This repository contains the code for predicting climate attitude using transformer-based models. The project utilizes the `transformers` library by Hugging Face to train and evaluate a sequence classification model on climate-related data.

To install the necessary dependencies, run the following commands:

```
pip install datasets
pip install transformers[torch]==4.30 accelerate sentencepiece -U
```

The dataset should be in JSON format with the following structure:

```json
[
    {
        "instruction": "The text to classify",
        "output": "label"
    },
    ...
]
```

The `output` field should contain one of the following labels: `neutral`, `risk`, or `opportunity`.

To train the model:

1. **Load and Process the Dataset:**

   Load your dataset and convert it into a format compatible with Hugging Face's `datasets` library.

   ```python
   import json
   import requests
   from datasets import Dataset, ClassLabel
   import pandas as pd

   def load_data(url):
       if url.startswith('http'):
           response = requests.get(url)
           data = response.json()
       else:
           with open(url, 'r') as file:
               data = json.load(file)
       return data

   data_url = "/content/data/train.json"  # Update this to the path of your dataset
   data = load_data(data_url)

   ```

2. **Tokenize the Dataset:**

   Tokenize the dataset using a pre-trained tokenizer.

   ```
   from transformers import AutoTokenizer

   def tokenize_and_encode(examples):
       tokenized_inputs = tokenizer(examples['instruction'], truncation=True, padding='max_length', max_length=128)
      
   tokenized_dataset = full_dataset.map(tokenize_and_encode, batched=True)
   ```

3. **Split the Dataset:**

   Split the dataset into training, validation, and testing sets.

   ```python
   from datasets import DatasetDict

   train_test_split = tokenized_dataset.train_test_split(test_size=0.1)  # 10% for testing
   train_val_split = train_test_split['train'].train_test_split(test_size=0.1)  # 10% of the remaining 90% for validation

   dataset_dict = DatasetDict({
       'train': train_val_split['train'],
       'validation': train_val_split['test'],
       'test': train_test_split['test']
   })
   ```

4. **Train the Model:**

   Define the training arguments and train the model.

   ```
   from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

   model = AutoModelForSequenceClassification.from_pretrained(model)

   training_args = TrainingArguments(
       output_dir='./results',
       num_train_epochs=30,
       evaluation_strategy='epoch',
       save_strategy='epoch'
   )


   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset_dict['train'],
       eval_dataset=dataset_dict['validation'],
       compute_metrics=compute_metrics
   )

   trainer.train()
   ```

5. **Save the Model:**


## To evaluate the model:

1. **Load the test dataset:**

   ```python
   data_url = "./data/test.json"  # Update this to the path of your dataset
   test_data = load_data(data_url)
   test_dataset = process_data_to_dataset(test_data)
   ```

2. **Tokenize the test dataset:**

   ```python
   test_dataset = test_dataset.map(tokenize_and_encode, batched=True)
   test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
   ```

3. **Evaluate the model:**

   ```
   from sklearn.metrics import accuracy_score, f1_score
   import torch
   from torch.utils.data import DataLoader

   def evaluate_model(model, dataset):
       model.eval()
       device = 'cuda' if torch.cuda.is_available() else 'cpu'
       model = model.to(device)
       data_loader = DataLoader(dataset, batch_size=16)
       predictions = []
       references = []

       for batch in data_loader:
           inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
           labels = batch['labels'].to(device)
           
           with torch.no_grad():
               outputs = model(**inputs)
           
           logits = outputs.logits
           preds = torch.argmax(logits, dim=1)
           
           predictions.extend(preds.cpu().numpy())
           references.extend(labels.cpu().numpy())

       accuracy = accuracy_score(references, predictions)
       f1 = f1_score(references, predictions, average='weighted')
       
       return {
           'accuracy': accuracy,
           'f1': f1
       }

   test_results = evaluate_model(model, test_dataset)
   print("Test Results:", test_results)
   ```

The model achieves the following results on the test dataset:
- **Accuracy**: [Accuracy value]
- **F1 Score**: [F1 Score value]

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


