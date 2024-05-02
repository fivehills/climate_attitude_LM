# climate_attitude_LM

https://huggingface.co/Kevintu/climate-attitude-LM


This language model is designed to assess the attitude expressed in texts about **climate change**. It categorizes the attitude into three types: risk, neutral, and opportunity. These categories correspond to the negative, neutral, and positive classifications commonly used in sentiment analysis. We employed a fine-tuning approach to adapt the final layer of the "cardiffnlp/twitter-roberta-base-sentiment-latest" model using a training dataset from "climatebert/climate_sentiment."

In comparison to similar existing models, such as "climatebert/distilroberta-base-climate-sentiment" and "XerOpred/twitter-climate-sentiment-model," which typically achieve accuracies ranging from 10% to 30% and F1 scores around 15%, our model demonstrates exceptional performance. When evaluated using the test dataset from "climatebert/climate_sentiment," it achieves an accuracy of 89% and an F1 score of 89%.

**Note** that you should paste or type a text concerning the **climate change** in the API input bar or using the testing code. Otherwise, the model does not work. e,.g, the example input, "The Deputy Mayor also said that while RRF regulations are clear at EU level, bureaucracy at national level in Italy has led to confusion for cities. “At national level, the implementation of RRF funds is not straightforward, there are ministries that have their own rules and cities get different information from different people,” she explained.
Adding to the discussion on whether a simpler, faster model, led by national governments, is best for dispersing EU funds, Boni stressed the importance of finding the right balance.“

Please cite: ``Sun., K, and Wang, R. 2024. The fine-tuned language model for detecting human attitudes to climate changes'' if you use this model.
   
  The following code shows how to test in the model.
 
  ```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "model"  # Ensure this path points to the correct directory
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the path to your text file
file_path = 'yourtext.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    new_text = file.read()

# Encode the text using the tokenizer used during training
encoded_input = tokenizer(new_text, return_tensors='pt', padding=True, truncation=True, max_length=64)

# Move the model to the correct device (CPU or GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)  # Move model to the correct device
encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # Move tensor to the correct device

model.eval()  # Set the model to evaluation mode

# Perform the prediction
with torch.no_grad():
    outputs = model(**encoded_input)

# Get the predictions (assumes classification with labels)
predictions = outputs.logits.squeeze()

# Assuming softmax is needed to interpret the logits as probabilities
probabilities = torch.softmax(predictions, dim=0)

# Define labels for each class index based on your classification categories
labels = ["risk", "neutral", "opportunity"]
predicted_index = torch.argmax(probabilities).item()  # Get the index of the max probability
predicted_label = labels[predicted_index]
predicted_probability = probabilities[predicted_index].item()

# Print the predicted label and its probability
print(f"Predicted Label: {predicted_label}, Probability: {predicted_probability:.4f}")

##the output example: predicted Label: neutral, Probability: 0.8377

```
