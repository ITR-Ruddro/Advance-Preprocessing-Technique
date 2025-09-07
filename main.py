import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch import nn
from preprocessing import PreprocessingPipeline

# 1. Configuration
DATASET_PATH = 'data/bangla_news_dataset.csv'
MODEL_DIR = 'models'
OUTPUT_DIR = 'results'
NUM_RUNS = 5
RANDOM_SEEDS = [42, 101, 202, 303, 404]

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 2. Data Loading and Splitting
def load_data(path):
    df = pd.read_csv(path)
    return df['text'], df['category']

X, y = load_data(DATASET_PATH)

# Use a fixed random seed for reproducible splits
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 3. Model Training and Evaluation Functions
def train_and_evaluate_ml_model(model, vectorizer, X_train, y_train, X_test, y_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'report': classification_report(y_test, y_pred)
    }

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding=True, max_length=512)
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(self.labels.iloc[idx])
        }

def train_and_evaluate_transformer(model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(y.unique()))

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(test_dataset.labels, y_pred),
        'f1_score': f1_score(test_dataset.labels, y_pred, average='macro'),
        'report': classification_report(test_dataset.labels, y_pred)
    }

# 4. Main Execution Loop for Statistical Analysis
def main():
    results = {
        'SVM': {'accuracy': [], 'f1_score': []},
        'RF': {'accuracy': [], 'f1_score': []},
        'BiLSTM': {'accuracy': [], 'f1_score': []},
        'BanglaBERT': {'accuracy': [], 'f1_score': []}
    }

    pipeline = PreprocessingPipeline()
    X_preprocessed = pipeline.run(X)

    for seed in RANDOM_SEEDS:
        # Re-split data for each run
        X_train, X_temp, y_train, y_temp = train_test_split(X_preprocessed, y, test_size=0.2, random_state=seed, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
        
        # ML Models
        tfidf_vectorizer = TfidfVectorizer()
        svm_model = SVC(kernel='linear')
        rf_model = RandomForestClassifier(n_estimators=100)
        
        svm_results = train_and_evaluate_ml_model(svm_model, tfidf_vectorizer, X_train, y_train, X_test, y_test)
        rf_results = train_and_evaluate_ml_model(rf_model, tfidf_vectorizer, X_train, y_train, X_test, y_test)
        
        results['SVM']['accuracy'].append(svm_results['accuracy'])
        results['SVM']['f1_score'].append(svm_results['f1_score'])
        results['RF']['accuracy'].append(rf_results['accuracy'])
        results['RF']['f1_score'].append(rf_results['f1_score'])

        # Transformer Models (replace with your specific model names)
        # Bilstm_results = train_and_evaluate_transformer('your_bilstm_model_name', X_train, y_train, X_val, y_val, X_test, y_test)
        # bert_results = train_and_evaluate_transformer('BanglaBERT-Base', X_train, y_train, X_val, y_val, X_test, y_test)

    # Print averaged results (mean and std)
    for model_name, metrics in results.items():
        print(f"--- {model_name} Results ---")
        mean_acc = np.mean(metrics['accuracy'])
        std_acc = np.std(metrics['accuracy'])
        mean_f1 = np.mean(metrics['f1_score'])
        std_f1 = np.std(metrics['f1_score'])
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == '__main__':
    main()
