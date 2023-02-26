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
