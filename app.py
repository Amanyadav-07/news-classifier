import os
import re
import logging
import json
import math
from collections import Counter, defaultdict
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
model = None
vocabulary = None
feature_weights = None

# Simple stopwords list (most common English stopwords)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'being', 'been',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would', 'should',
    'could', 'ought', 'i\'m', 'you\'re', 'he\'s', 'she\'s', 'it\'s', 'we\'re', 'they\'re'
}

# Categories for classification
CATEGORIES = ['Politics', 'Sports', 'Business', 'Technology', 'Entertainment']

def simple_tokenize(text):
    """Simple tokenization without external dependencies."""
    # Convert to lowercase and extract words
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    # Remove stopwords and short words
    words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return words

def clean_text(text):
    """Clean and preprocess text data."""
    try:
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and clean
        tokens = simple_tokenize(text)
        
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in text cleaning: {str(e)}")
        return ""

class SimpleNaiveBayes:
    """Simple Naive Bayes classifier implementation."""
    
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.classes = set()
        self.total_docs = 0
        
    def fit(self, texts, labels):
        """Train the classifier."""
        # Count documents per class
        for label in labels:
            self.class_counts[label] += 1
            self.classes.add(label)
        
        self.total_docs = len(labels)
        
        # Count features per class
        for text, label in zip(texts, labels):
            words = simple_tokenize(text)
            for word in words:
                self.feature_counts[label][word] += 1
                self.vocabulary.add(word)
        
        # Calculate feature probabilities
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        for label in self.classes:
            total_words_in_class = sum(self.feature_counts[label].values())
            vocab_size = len(self.vocabulary)
            
            for word in self.vocabulary:
                # Laplace smoothing
                count = self.feature_counts[label][word]
                self.feature_probs[label][word] = (count + 1) / (total_words_in_class + vocab_size)
    
    def predict_proba(self, texts):
        """Predict class probabilities."""
        results = []
        
        for text in texts:
            words = simple_tokenize(text)
            class_scores = {}
            
            for label in self.classes:
                # Calculate log probability
                log_prob = math.log(self.class_counts[label] / self.total_docs)
                
                for word in words:
                    if word in self.vocabulary:
                        log_prob += math.log(self.feature_probs[label][word])
                
                class_scores[label] = log_prob
            
            # Convert to probabilities using softmax
            max_score = max(class_scores.values())
            exp_scores = {label: math.exp(score - max_score) for label, score in class_scores.items()}
            total_exp = sum(exp_scores.values())
            
            probabilities = {label: exp_score / total_exp for label, exp_score in exp_scores.items()}
            results.append(probabilities)
        
        return results
    
    def predict(self, texts):
        """Predict classes."""
        probabilities = self.predict_proba(texts)
        predictions = []
        
        for prob_dict in probabilities:
            predicted_class = max(prob_dict.keys(), key=lambda k: prob_dict[k])
            predictions.append(predicted_class)
        
        return predictions

def create_sample_data():
    """Create sample news data for training the model."""
    sample_data = [
        # Politics - expanded dataset
        ("Government announces new policy changes for healthcare reform and social security", "Politics"),
        ("Presidential election results show tight race between candidates with voter turnout", "Politics"),
        ("Senate votes on infrastructure bill with bipartisan support and amendments", "Politics"),
        ("Mayor discusses city budget tax reforms and municipal spending priorities", "Politics"),
        ("International trade agreements signed between countries for economic cooperation", "Politics"),
        ("Congress debates immigration reform legislation with border security measures", "Politics"),
        ("Supreme court ruling affects constitutional rights and legal precedents", "Politics"),
        ("Political campaign fundraising events draw supporters and media attention", "Politics"),
        ("Diplomatic negotiations between nations seek peaceful resolution of conflicts", "Politics"),
        ("Legislative session addresses environmental policy and climate change initiatives", "Politics"),
        
        # Sports - expanded dataset
        ("Football championship game ends in overtime thriller with record attendance", "Sports"),
        ("Basketball player breaks scoring record in season finale performance", "Sports"),
        ("Olympic games preparation underway with new training facilities and athletes", "Sports"),
        ("Tennis tournament upset victory by unseeded player surprises spectators", "Sports"),
        ("Baseball season kicks off with opening day celebrations and ceremonies", "Sports"),
        ("Soccer world cup qualifiers feature exciting matches and international competition", "Sports"),
        ("Swimming championship records broken by talented athletes in pool events", "Sports"),
        ("Golf tournament professional players compete for major championship title", "Sports"),
        ("Hockey playoffs feature intense games with skilled players and goalkeepers", "Sports"),
        ("Track and field events showcase athletic performance and world records", "Sports"),
        
        # Business - expanded dataset
        ("Stock market reaches all-time high amid strong quarterly earnings reports", "Business"),
        ("Technology company reports record profits and revenue growth this quarter", "Business"),
        ("Startup secures million dollar investment funding from venture capital firms", "Business"),
        ("Corporate merger between major companies creates industry leading conglomerate", "Business"),
        ("Economic indicators show signs of recovery and sustained market growth", "Business"),  
        ("Banking sector announces new financial services and digital transformation", "Business"),
        ("Retail sales increase during holiday shopping season with consumer spending", "Business"),
        ("Manufacturing industry expands production capacity and global supply chains", "Business"),
        ("Real estate market shows price appreciation and increased property transactions", "Business"),
        ("Cryptocurrency trading volume surges with blockchain technology adoption", "Business"),
        
        # Technology - expanded dataset
        ("Artificial intelligence breakthrough enables medical diagnosis and treatment advances", "Technology"),
        ("Smartphone features revolutionary camera technology and processing capabilities", "Technology"),
        ("Software update brings enhanced security features and performance improvements", "Technology"),
        ("Quantum computing advances promise faster data processing and calculations", "Technology"),
        ("Social media platform introduces new privacy features and user controls", "Technology"),
        ("Cloud computing services expand with improved infrastructure and reliability", "Technology"),
        ("Cybersecurity measures protect against digital threats and data breaches", "Technology"),
        ("Virtual reality applications transform gaming entertainment and education experiences", "Technology"),
        ("Internet connectivity improves with fiber optic networks and bandwidth expansion", "Technology"),
        ("Machine learning algorithms enhance data analysis and predictive modeling capabilities", "Technology"),
        
        # Entertainment - expanded dataset
        ("Movie premiere draws celebrity crowds and red carpet media coverage", "Entertainment"),
        ("Music festival lineup features popular artists and live performance acts", "Entertainment"),
        ("Television series wins multiple awards at annual ceremony celebration", "Entertainment"),
        ("Bestselling book adaptation planned for upcoming movie production release", "Entertainment"),
        ("Gaming convention showcases latest video game releases and interactive technology", "Entertainment"),
        ("Concert tour announcement generates excitement among fans and ticket sales", "Entertainment"),
        ("Theater production receives critical acclaim and audience standing ovations", "Entertainment"),
        ("Celebrity wedding ceremony attracts media attention and public interest", "Entertainment"),
        ("Documentary film explores important social issues and cultural themes", "Entertainment"),
        ("Comedy show features popular comedians and hilarious stand-up performances", "Entertainment")
    ]
    
    return sample_data

def train_model():
    """Train the news classification model."""
    try:
        logger.info("Creating sample data and training model...")
        
        # Create sample data
        data = create_sample_data()
        texts, labels = zip(*data)
        
        # Clean texts
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Create and train simple classifier
        classifier = SimpleNaiveBayes()
        classifier.fit(cleaned_texts, labels)
        
        # Test accuracy on training data (for demonstration)
        predictions = classifier.predict(cleaned_texts)
        correct = sum(1 for pred, actual in zip(predictions, labels) if pred == actual)
        accuracy = correct / len(labels)
        
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")
        logger.info(f"Training data size: {len(labels)} samples")
        logger.info(f"Vocabulary size: {len(classifier.vocabulary)} words")
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def predict_category(text, confidence_threshold=0.3):
    """Predict category for given text with confidence score."""
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
        
        # Clean the input text
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            raise ValueError("Text becomes empty after cleaning")
        
        # Make prediction
        predictions = model.predict([cleaned_text])
        prediction = predictions[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = max(probabilities.values())
        
        return {
            'category': prediction,
            'confidence': float(confidence),
            'all_probabilities': {k: float(v) for k, v in probabilities.items()},
            'high_confidence': confidence >= confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

# Initialize the model at startup
try:
    model = train_model()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    model = None

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Web form prediction route."""
    try:
        if model is None:
            return render_template('index.html', error="Model not available. Please try again later.")
        
        text = request.form.get('news_text', '').strip()
        
        if not text:
            return render_template('index.html', error="Please enter some news text to classify.")
        
        if len(text) < 10:
            return render_template('index.html', error="Please enter at least 10 characters.")
        
        # Get prediction
        result = predict_category(text)
        
        return render_template('index.html', 
                             prediction=result['category'],
                             confidence=result['confidence'],
                             all_probabilities=result['all_probabilities'],
                             high_confidence=result['high_confidence'],
                             input_text=text)
        
    except ValueError as e:
        return render_template('index.html', error=str(e))
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return render_template('index.html', error="An error occurred during prediction. Please try again.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for prediction."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not available',
                'status': 'error'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'status': 'error'
            }), 400
        
        if len(text) < 10:
            return jsonify({
                'error': 'Text must be at least 10 characters long',
                'status': 'error'
            }), 400
        
        # Get prediction
        result = predict_category(text)
        
        return jsonify({
            'status': 'success',
            'prediction': result['category'],
            'confidence': result['confidence'],
            'all_probabilities': result['all_probabilities'],
            'high_confidence': result['high_confidence'],
            'input_text': text
        })
        
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    except Exception as e:
        logger.error(f"Error in API predict route: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_available': model is not None,
        'categories': CATEGORIES
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', error="Internal server error. Please try again."), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)