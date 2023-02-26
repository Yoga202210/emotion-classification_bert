# emotion-classification_bert
## install useful libraries
```pip install transformers```

```
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
```
## read csv dataset
```
train_df = pd.read_csv("/kaggle/input/datafiles/train_data.csv",header=None)
train_df.columns = ["Text","Emotion"]
train_df.head()
```
![1.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/1.png?raw=true)
```
train_df['Emotion'].value_counts()
```
![2.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/2.png?raw=true)
```
val_df = pd.read_csv("/kaggle/input/datafiles/val_data.csv",header=None)
val_df.columns = ["Text","Emotion"]
val_df.head()
```
![3.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/3.png?raw=true)
```
val_df['Emotion'].value_counts()
```
![4.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/4.png?raw=true)
## convert text labels into numeric labels
```
data_dict = {0:'joy',1:'sadness',2:'anger',3:'fear',4:'love',5:'surprise'}
my_dict = { data_dict[k]:k for k in data_dict}
train_df['Emotion'] = [my_dict.get(i,i) for i in list(train_df['Emotion'])]
train_df.head()
```
![5.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/5.png?raw=true)
```
train_df['Emotion'].value_counts()
```
![6.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/6.png?raw=true)
```
val_df['Emotion'] = [my_dict.get(i,i) for i in list(val_df['Emotion'])]
val_df.head()
```
![7.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/7.png?raw=true)
```
val_df['Emotion'].value_counts()
```
![8.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/8.png?raw=true)
## load data using datasets library
```
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
```
```
data_files = {'train': "train.csv",
              'val': "val.csv"}
```
```
dataset = load_dataset('csv', data_files=data_files)
```
```
dataset
```
![9.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/9.png?raw=true)
## use Tokenizer to prepare dataset
```
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
```
def tokenize_function(example):
    return tokenizer(example["Text"], truncation=True)
```
```
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
```
tokenized_datasets
```
![10.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/10.png?raw=true)
```
tokenized_datasets = tokenized_datasets.remove_columns(["Text"])
tokenized_datasets = tokenized_datasets.rename_column("Emotion", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```
![11.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/11.png?raw=true)
```
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
```
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=4, collate_fn=data_collator
)
val_dataloader = DataLoader(
    tokenized_datasets["val"], batch_size=4, collate_fn=data_collator
)
```
## use bert model to train dataset
```
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
```
```
optimizer = AdamW(model.parameters(), lr=5e-5)
```
```
num_epochs = 4
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)
```
16000
```
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device
```
device(type='cuda')
## training
```
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
## validation
```
val = []
val_pred = []
model.eval()
for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    val_pred = val_pred + (outputs.logits.argmax(axis=-1).flatten().tolist())
    val = val + batch['labels'].tolist()
```
```
val = [data_dict.get(i,i) for i in val]
val_pred = [data_dict.get(i,i) for i in val_pred]
```
```
cr_val = classification_report(val,val_pred)
val_accuracy = accuracy_score(val,val_pred)
print("Validation accuracy:", val_accuracy)
print(cr_val)
```
![12.png](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/12.png?raw=true)
## test data prediction
```
with open("test_prediction.csv","w",encoding="utf-8") as f1:
    with open ("/kaggle/input/train-data/test_data.txt","r",encoding="utf-8") as f2:
        for line in f2:
            single_tokenized_test = tokenizer(line,truncation=True, return_tensors='pt')
            with torch.no_grad():
                single_tokenized_test = {k: v.to(device) for k, v in single_tokenized_test.items()}
                output = model(**single_tokenized_test)
                single_test_pred = output.logits.argmax(axis=-1).flatten().tolist()
                f1.write(str(data_dict[single_test_pred[0]])+"\n")
```
## UI system (a corresponding emoji to show the predicted emotion)
```
pip install emoji
```
```
from ipywidgets import widgets
lbl1=widgets.Label("Input Sentence:")
display(lbl1)
text=widgets.Text()
display(text)
btn=widgets.Button(description="The predicted emotion")
display(btn)
lbl2=widgets.Label()
display(lbl2)
emotion_dict = {0:'\U0001F601',1:'\U0001F62D',2:'\U0001F621',3:'\U0001F631',4:'\U0001F60D',5:'\U0001F632'}
def predictedemotion(b):
    inp=text.value
    single_tokenized_test = tokenizer(inp,truncation=True, return_tensors='pt')
    with torch.no_grad():
        single_tokenized_test = {k: v.to(device) for k, v in single_tokenized_test.items()}
        output = model(**single_tokenized_test)
        single_test_pred = output.logits.argmax(axis=-1).flatten().tolist()
        lbl2.value=emotion_dict[single_test_pred[0]]
btn.on_click(predictedemotion)
```
![13.jpeg](https://github.com/Yoga202210/emotion-classification_bert/blob/main/images/13.jpeg?raw=true)
















