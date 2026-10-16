# import json
# import random
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# import torch
# from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
# from flask import Flask, render_template, request, jsonify

# # Load intents data
# with open('intents.json', encoding='utf-8') as file:
#     intents = json.load(file)

# # Prepare data
# sentences = []
# labels = []
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         sentences.append(pattern)
#         labels.append(intent['tag'])

# # Convert labels to numbers
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(labels)

# # Split data
# train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.3, random_state=42, stratify=labels)

# # Load BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def tokenize(sentences, labels, max_length=50):
#     input_ids, attention_masks = [], []
#     for sentence in sentences:
#         encoded_dict = tokenizer.encode_plus(
#             sentence,
#             add_special_tokens=True,
#             max_length=max_length,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#             truncation=True  # Explicitly activate truncation
#         )
#         input_ids.append(encoded_dict['input_ids'])
#         attention_masks.append(encoded_dict['attention_mask'])
#     return input_ids, attention_masks, labels

# train_inputs, train_masks, train_labels = tokenize(train_sentences, train_labels)
# val_inputs, val_masks, val_labels = tokenize(val_sentences, val_labels)

# # Load BERT model
# model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased',
#     num_labels=len(label_encoder.classes_),
#     output_attentions=False,
#     output_hidden_states=False,
# )

# # Prepare optimizer
# optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# # Prepare DataLoader
# batch_size = 32

# train_data = TensorDataset(torch.cat(train_inputs, dim=0), torch.cat(train_masks, dim=0), torch.tensor(train_labels))
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# validation_data = TensorDataset(torch.cat(val_inputs, dim=0), torch.cat(val_masks, dim=0), torch.tensor(val_labels))
# validation_sampler = SequentialSampler(validation_data)
# validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# # Training loop
# epochs = 6

# for epoch in range(epochs):
#     model.train()
#     total_loss = 0

#     for step, batch in enumerate(train_dataloader):
#         batch_input_ids, batch_input_mask, batch_labels = batch
#         model.zero_grad()
#         outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask, labels=batch_labels)
#         loss = outputs[0]
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()

#     avg_train_loss = total_loss / len(train_dataloader)

#     print(f'Epoch {epoch+1}/{epochs}')
#     print(f'Average training loss: {avg_train_loss:.2f}')

#     model.eval()

#     eval_loss = 0
#     eval_accuracy = 0

#     for batch in validation_dataloader:
#         batch_input_ids, batch_input_mask, batch_labels = batch

#         with torch.no_grad():
#             outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

#         logits = outputs[0]
#         logits = logits.detach().cpu().numpy()
#         label_ids = batch_labels.to('cpu').numpy()

#         eval_accuracy += np.sum(np.argmax(logits, axis=1) == label_ids)

#     print(f'Validation accuracy: {eval_accuracy / len(val_labels):.2f}')

# # Flask app
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/get_response', methods=['POST'])
# def get_bot_response():
#     data = request.get_json()
#     user_message = data['user_message']
    
#     inputs = tokenizer.encode_plus(
#         user_message,
#         add_special_tokens=True,
#         max_length=50,
#         pad_to_max_length=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#         truncation=True  # Explicitly activate truncation
#     )
    
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
    
#     with torch.no_grad():
#         outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    
#     logits = outputs[0]
#     predicted_label = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]
#     intent = label_encoder.inverse_transform([predicted_label])[0]
    
#     for i in intents['intents']:
#         if i['tag'] == intent:
#             response = random.choice(i['responses'])
#             break
    
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)


import json
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, render_template, request, jsonify

# Load intents data
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare data
sentences = []
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Convert labels to numbers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all sentences directly for training
encoded_inputs = tokenizer(
    sentences,
    padding='max_length',
    max_length=32,
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
labels_tensor = torch.tensor(labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_),
    output_attentions=False,
    output_hidden_states=False,
)

# Prepare optimizer
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)

# Prepare DataLoader (batch_size=4 ممتاز للبيانات القليلة)
dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=4)

# Training loop
epochs = 15
print("🚀 جاري تدريب النموذج...")

model.train()
for epoch in range(epochs):
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch_input_ids, batch_input_mask, batch_labels = batch
        model.zero_grad()
        
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask, labels=batch_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{epochs} - Average training loss: {avg_train_loss:.4f}')

print("✅ اكتمل التدريب بنجاح!")
model.eval()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    data = request.get_json()
    user_message = data['user_message']
    
    inputs = tokenizer(
        user_message,
        padding='max_length',
        max_length=32,
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    
    logits = outputs[0]
    predicted_label = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]
    intent = label_encoder.inverse_transform([predicted_label])[0]
    
    response = "عذراً، لم أفهم سؤالك."
    for i in intents['intents']:
        if i['tag'] == intent:
            response = random.choice(i['responses'])
            break
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)