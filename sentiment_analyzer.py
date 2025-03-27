import pandas as pd
import numpy as np
import re
import nltk
import joblib
import logging
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripAdvisorSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Create a pipeline with TF-IDF and Naive Bayes
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('clf', MultinomialNB())
        ])
        
        # Define parameter grid for GridSearchCV
        self.param_grid = {
            'tfidf__max_features': [3000, 5000, 7000],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__min_df': [1, 2, 3],
            'tfidf__max_df': [0.9, 0.95, 0.99],
            'clf__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        
        # Define scoring metrics
        self.scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Tokenization
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def convert_ratings_to_sentiment(self, rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    def prepare_data(self, df):
        # Preprocess review texts
        logger.info("Preprocessing review texts...")
        df['processed_text'] = df['review_text'].apply(self.preprocess_text)
        # Convert ratings to sentiment labels
        df['sentiment'] = df['rating'].apply(self.convert_ratings_to_sentiment)
        return df
    
    def train_model(self, X_train, y_train):
        logger.info("Training model with GridSearchCV...")
        
        # Perform k-fold cross-validation with multiple metrics
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=kfold,
            n_jobs=-1,
            verbose=1,
            scoring=self.scoring,
            refit='f1_macro'
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Update pipeline with best parameters
        self.pipeline = grid_search.best_estimator_
        
        # Log results
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info("Cross-validation scores for best model:")
        for metric in self.scoring:
            mean_score = grid_search.cv_results_[f'mean_test_{metric}'][grid_search.best_index_]
            std_score = grid_search.cv_results_[f'std_test_{metric}'][grid_search.best_index_]
            logger.info(f"{metric}: {mean_score:.3f} (+/- {std_score*2:.3f})")
        
        return grid_search.best_score_
    
    def evaluate_model(self, X_test, y_test):
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate detailed metrics
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Calculate and print class-wise metrics
        classes = self.pipeline.classes_
        class_metrics = {}
        
        for i, cls in enumerate(classes):
            true_pos = conf_matrix[i, i]
            false_pos = conf_matrix[:, i].sum() - true_pos
            false_neg = conf_matrix[i, :].sum() - true_pos
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        print("\nDetailed Class-wise Metrics:")
        for cls, metrics in class_metrics.items():
            print(f"\nClass: {cls}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-score: {metrics['f1_score']:.3f}")
    
    def predict_sentiment(self, text):
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        # Make prediction
        sentiment = self.pipeline.predict([processed_text])[0]
        # Get probability scores
        proba = self.pipeline.predict_proba([processed_text])[0]
        confidence = max(proba)
        return sentiment, confidence
    
    def save_model(self, filepath='models/sentiment_analyzer_model.joblib'):
        """Save the trained model to disk"""
        try:
            joblib.dump(self.pipeline, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath='models/sentiment_analyzer_model.joblib'):
        """Load a trained model from disk"""
        try:
            self.pipeline = joblib.load(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

def main():
    # Load sample reviews
    logger.info("Loading sample reviews...")
    df = pd.read_csv('sample_reviews.csv')
    
    # Initialize analyzer
    analyzer = TripAdvisorSentimentAnalyzer()
    
    # Prepare data
    df = analyzer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )
    
    # Train and evaluate model
    best_score = analyzer.train_model(X_train, y_train)
    
    # Evaluate on test set
    analyzer.evaluate_model(X_test, y_test)
    
    # Save the trained model
    analyzer.save_model()
    
    # Test with some new reviews
    test_reviews = [
        "The hotel was beautiful but the service could have been better",
        "Absolutely terrible experience, would never come back",
        "Great location, amazing staff, and wonderful amenities!",
        "The room was clean but the wifi was slow and the breakfast was average",
        "Outstanding luxury experience with impeccable service and stunning views"
    ]
    
    print("\nSample Predictions:")
    for review in test_reviews:
        sentiment, confidence = analyzer.predict_sentiment(review)
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
