"""
Travel Assistant NLP model with improved intent classification, sentiment analysis, and NER
"""
import os
import torch
import numpy as np
import spacy
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Any, Tuple
import nltk
from datasets import Dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.sentiment_analyzer import TripAdvisorSentimentAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TravelAssistantNLP:
    def __init__(self):
        """Initialize the model with custom configurations"""
        self.intent_labels = [
            'book_flight',
            'hotel_booking',
            'restaurant_recommendation',
            'tourist_attraction',
            'transportation'
        ]
        
        # Load spaCy model with custom entity patterns
        self.nlp = spacy.load("en_core_web_sm")
        
        # Add custom entity patterns for better NER
        ruler = self.nlp.get_pipe("entity_ruler") if "entity_ruler" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "FAC", "pattern": [{"LOWER": "jfk"}, {"LOWER": "airport"}]},
            {"label": "FAC", "pattern": [{"LOWER": "heathrow"}, {"LOWER": "airport"}]},
            {"label": "ORG", "pattern": "Hilton"},
            {"label": "ORG", "pattern": "Marriott"},
            {"label": "ORG", "pattern": "Ritz Carlton"},
            {"label": "FAC", "pattern": "Eiffel Tower"},
            {"label": "FAC", "pattern": "Times Square"},
            {"label": "FAC", "pattern": "Sagrada Familia"}
        ]
        ruler.add_patterns(patterns)
        
        # Initialize sentiment analyzer with improved thresholds
        self.sentiment_analyzer = TripAdvisorSentimentAnalyzer()
        self.sentiment_threshold = 0.6
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(self.intent_labels)
        )

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better model understanding"""
        doc = self.nlp(text.lower().strip())
        return " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])

    def predict(self, text: str) -> Dict[str, Any]:
        """Make predictions with improved confidence handling"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get intent prediction
        inputs = self.tokenizer(
            processed_text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
            
        intent = self.intent_labels[predicted.item()]
        confidence = confidence.item()
        
        # Get entities with improved NER
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Get sentiment with better confidence handling
        sentiment_result = self.sentiment_analyzer.predict(text)
        sentiment_label = sentiment_result[0]
        sentiment_score = sentiment_result[1]
        
        return {
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "sentiment": (sentiment_label, sentiment_score)
        }

    def train(self, train_texts: List[str], train_labels: List[str], 
              eval_texts: List[str] = None, eval_labels: List[str] = None,
              output_dir: str = "models/intent_classifier"):
        """Train the model with improved parameters"""
        # Convert string labels to indices
        train_label_ids = [self.intent_labels.index(label) for label in train_labels]
        eval_label_ids = [self.intent_labels.index(label) for label in eval_labels] if eval_labels else None
        
        # Calculate class weights for balanced training
        label_counts = np.bincount(train_label_ids)
        if len(label_counts) < len(self.intent_labels):
            label_counts = np.pad(label_counts, (0, len(self.intent_labels) - len(label_counts)))
        total = len(train_labels)
        class_weights = torch.FloatTensor([total / (len(self.intent_labels) * c) if c > 0 else 1.0 for c in label_counts])
        
        # Prepare datasets
        train_encodings = self.tokenizer(
            [self.preprocess_text(text) for text in train_texts],
            truncation=True,
            padding=True
        )
        
        train_dataset = Dataset.from_dict({
            **train_encodings,
            'labels': train_label_ids
        })
        
        if eval_texts and eval_labels:
            eval_encodings = self.tokenizer(
                [self.preprocess_text(text) for text in eval_texts],
                truncation=True,
                padding=True
            )
            eval_dataset = Dataset.from_dict({
                **eval_encodings,
                'labels': eval_label_ids
            })
        else:
            eval_dataset = None
        
        # Training arguments with improved parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            load_best_model_at_end=True,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            learning_rate=2e-5
        )
        
        # Custom trainer with weighted loss
        class WeightedTrainer(Trainer):
            def __init__(self, num_labels, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.num_labels = num_labels
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Compute weighted loss
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss

        # Initialize and train
        trainer = WeightedTrainer(
            num_labels=len(self.intent_labels),
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
