#!/usr/bin/env python3
"""
Simple HTTP server for news classification without external dependencies.
"""

import http.server
import socketserver
import urllib.parse
import json
import os
import re
import logging
import math
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

CATEGORIES = ['Politics', 'Sports', 'Business', 'Technology', 'Entertainment']

def simple_tokenize(text):
    """Simple tokenization without external dependencies."""
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return words

def clean_text(text):
    """Clean and preprocess text data."""
    try:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
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
        for label in labels:
            self.class_counts[label] += 1
            self.classes.add(label)
        
        self.total_docs = len(labels)
        
        for text, label in zip(texts, labels):
            words = simple_tokenize(text)
            for word in words:
                self.feature_counts[label][word] += 1
                self.vocabulary.add(word)
        
        self.feature_probs = defaultdict(lambda: defaultdict(float))
        for label in self.classes:
            total_words_in_class = sum(self.feature_counts[label].values())
            vocab_size = len(self.vocabulary)
            
            for word in self.vocabulary:
                count = self.feature_counts[label][word]
                self.feature_probs[label][word] = (count + 1) / (total_words_in_class + vocab_size)
    
    def predict_proba(self, texts):
        """Predict class probabilities."""
        results = []
        
        for text in texts:
            words = simple_tokenize(text)
            class_scores = {}
            
            for label in self.classes:
                log_prob = math.log(self.class_counts[label] / self.total_docs)
                
                for word in words:
                    if word in self.vocabulary:
                        log_prob += math.log(self.feature_probs[label][word])
                
                class_scores[label] = log_prob
            
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
        
        data = create_sample_data()
        texts, labels = zip(*data)
        
        cleaned_texts = [clean_text(text) for text in texts]
        
        classifier = SimpleNaiveBayes()
        classifier.fit(cleaned_texts, labels)
        
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

def predict_category(text, model, confidence_threshold=0.3):
    """Predict category for given text with confidence score."""
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text")
        
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            raise ValueError("Text becomes empty after cleaning")
        
        predictions = model.predict([cleaned_text])
        prediction = predictions[0]
        
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

# Initialize the model
classifier_model = train_model()

class NewsClassifierHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the news classifier."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.serve_index()
        elif self.path == '/health':
            self.serve_health()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/predict':
            self.handle_form_predict()
        elif self.path == '/api/predict':
            self.handle_api_predict()
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve the main HTML page."""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Classifier - AI-Powered News Categorization</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            position: relative;
            overflow-x: hidden;
        }

        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
        }

        .main-container {
            position: relative;
            z-index: 1;
            padding: 2rem 0;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: none;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        }

        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px 20px 0 0 !important;
            border: none;
            text-align: center;
            padding: 2rem;
        }

        .card-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .card-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .form-control {
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.9);
        }

        .form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
            background: white;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }

        .alert {
            border: none;
            border-radius: 15px;
            padding: 1.5rem;
            font-size: 1rem;
            margin-top: 1.5rem;
            animation: slideIn 0.5s ease-out;
        }

        .alert-success {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            color: #2d5016;
        }

        .alert-danger {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
            color: #721c24;
        }

        .prediction-result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }

        .prediction-category {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .confidence-score {
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            opacity: 0.9;
        }

        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 10px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }

        .confidence-fill {
            background: rgba(255, 255, 255, 0.8);
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease-out;
            animation: fillBar 1s ease-out;
        }

        .probabilities {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
        }

        .probability-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }

        .probability-item:last-child {
            margin-bottom: 0;
        }

        .probability-bar {
            width: 100px;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-left: 1rem;
        }

        .probability-fill {
            height: 100%;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 4px;
            transition: width 1s ease-out;
        }

        .api-section {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            color: white;
        }

        .code-block {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .category-icons {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        .footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 3rem;
            padding: 2rem;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fillBar {
            from {
                width: 0%;
            }
        }

        @media (max-width: 768px) {
            .card-header h1 {
                font-size: 2rem;
            }
            
            .prediction-category {
                font-size: 2rem;
            }
            
            .category-icons {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div id="particles-js"></div>
    
    <div class="container main-container">
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="card">
                    <div class="card-header">
                        <h1><i class="fas fa-newspaper"></i> News Classifier</h1>
                        <p>AI-Powered News Categorization using NLP & Machine Learning</p>
                    </div>
                    <div class="card-body p-4">
                        <form method="POST" action="/predict" id="newsForm">
                            <div class="mb-4">
                                <label for="news_text" class="form-label">
                                    <i class="fas fa-edit"></i> Enter News Article Text
                                </label>
                                <textarea 
                                    class="form-control" 
                                    id="news_text" 
                                    name="news_text" 
                                    rows="6" 
                                    placeholder="Paste your news article here... (minimum 10 characters)"
                                    required></textarea>
                                <div class="form-text">
                                    Categories: Politics, Sports, Business, Technology, Entertainment
                                </div>
                            </div>
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary" id="submitBtn">
                                    <i class="fas fa-magic" id="submitIcon"></i>
                                    <span id="submitText">Classify News</span>
                                </button>
                            </div>
                        </form>

                        <div class="api-section">
                            <h5><i class="fas fa-code me-2"></i>API Usage</h5>
                            <p>You can also use our JSON API endpoint for programmatic access:</p>
                            <div class="code-block">
POST /api/predict
Content-Type: application/json

{
  "text": "Your news article text here..."
}
                            </div>
                            <p><strong>Response:</strong></p>
                            <div class="code-block">
{
  "status": "success",
  "prediction": "Technology",
  "confidence": 0.85,
  "all_probabilities": {...},
  "high_confidence": true
}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>
                <i class="fas fa-heart text-danger"></i> 
                Built with Python and AI magic (No external ML libraries!)
            </p>
            <p>
                <small>
                    <i class="fas fa-info-circle me-1"></i>
                    This model uses a custom Naive Bayes implementation
                </small>
            </p>
        </div>
    </div>

    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/particles.js/2.0.0/particles.min.js"></script>
    
    <script>
        // Initialize particles.js
        particlesJS('particles-js', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: '#ffffff' },
                shape: { type: 'circle' },
                opacity: { value: 0.5, random: false },
                size: { value: 3, random: true },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#ffffff',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 6,
                    direction: 'none',
                    random: false,
                    straight: false,
                    out_mode: 'out',
                    bounce: false
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                },
                modes: {
                    grab: { distance: 400, line_linked: { opacity: 1 } },
                    bubble: { distance: 400, size: 40, duration: 2, opacity: 8, speed: 3 },
                    repulse: { distance: 200, duration: 0.4 },
                    push: { particles_nb: 4 },
                    remove: { particles_nb: 2 }
                }
            },
            retina_detect: true
        });

        // Character counter
        const textarea = document.getElementById('news_text');
        const createCharCounter = () => {
            const counter = document.createElement('div');
            counter.className = 'form-text text-end';
            counter.id = 'charCounter';
            textarea.parentNode.insertBefore(counter, textarea.nextSibling);
            return counter;
        };

        const charCounter = createCharCounter();
        
        const updateCharCounter = () => {
            const length = textarea.value.length;
            charCounter.textContent = `${length} characters`;
            
            if (length < 10) {
                charCounter.className = 'form-text text-end text-warning';
            } else {
                charCounter.className = 'form-text text-end text-success';
            }
        };

        textarea.addEventListener('input', updateCharCounter);
        updateCharCounter();

        // Auto-resize textarea
        const autoResize = () => {
            textarea.style.height = 'auto';
            textarea.style.height = (textarea.scrollHeight) + 'px';
        };

        textarea.addEventListener('input', autoResize);
        autoResize();
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_health(self):
        """Serve health check endpoint."""
        health_data = {
            'status': 'healthy',
            'model_available': classifier_model is not None,
            'categories': CATEGORIES
        }
        
        response = json.dumps(health_data)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode())
    
    def handle_form_predict(self):
        """Handle form-based prediction requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = urllib.parse.parse_qs(post_data)
            
            text = params.get('news_text', [''])[0].strip()
            
            if not text:
                self.serve_error_page("Please enter some news text to classify.")
                return
            
            if len(text) < 10:
                self.serve_error_page("Please enter at least 10 characters.")
                return
            
            # Get prediction
            result = predict_category(text, classifier_model)
            self.serve_result_page(result, text)
            
        except Exception as e:
            logger.error(f"Error in form predict: {str(e)}")
            self.serve_error_page("An error occurred during prediction. Please try again.")
    
    def handle_api_predict(self):
        """Handle API prediction requests."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data)
            except json.JSONDecodeError:
                self.send_json_error('Invalid JSON data', 400)
                return
            
            text = data.get('text', '').strip()
            
            if not text:
                self.send_json_error('No text provided', 400)
                return
            
            if len(text) < 10:
                self.send_json_error('Text must be at least 10 characters long', 400)
                return
            
            # Get prediction
            result = predict_category(text, classifier_model)
            
            response_data = {
                'status': 'success',
                'prediction': result['category'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities'],
                'high_confidence': result['high_confidence'],
                'input_text': text
            }
            
            response = json.dumps(response_data)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Content-length', str(len(response)))
            self.end_headers()
            self.wfile.write(response.encode())
            
        except Exception as e:
            logger.error(f"Error in API predict: {str(e)}")
            self.send_json_error('Internal server error', 500)
    
    def send_json_error(self, message, status_code):
        """Send JSON error response."""
        error_data = {
            'error': message,
            'status': 'error'
        }
        
        response = json.dumps(error_data)
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Content-length', str(len(response)))
        self.end_headers()
        self.wfile.write(response.encode())
    
    def serve_error_page(self, error_message):
        """Serve error page with the same styling as main page."""
        html_content = self.get_base_html_with_content(f'''
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                {error_message}
            </div>
        ''')
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_result_page(self, result, input_text):
        """Serve result page with prediction."""
        category = result['category']
        confidence = result['confidence']
        all_probabilities = result['all_probabilities']
        high_confidence = result['high_confidence']
        
        # Category icons
        icons = {
            'Politics': 'fas fa-landmark',
            'Sports': 'fas fa-football-ball',
            'Business': 'fas fa-chart-line',
            'Technology': 'fas fa-microchip',
            'Entertainment': 'fas fa-film'
        }
        
        icon = icons.get(category, 'fas fa-newspaper')
        confidence_icon = 'fas fa-check-circle' if high_confidence else 'fas fa-exclamation-circle'
        
        # Create probability bars
        prob_html = ''
        for cat, prob in all_probabilities.items():
            prob_html += f'''
                <div class="probability-item">
                    <span>{cat}</span>
                    <div class="d-flex align-items-center">
                        <span class="me-2">{prob * 100:.1f}%</span>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: {prob * 100}%"></div>
                        </div>
                    </div>
                </div>
            '''
        
        content = f'''
            <div class="prediction-result">
                <div class="category-icons">
                    <i class="{icon}"></i>
                </div>
                <div class="prediction-category">{category}</div>
                <div class="confidence-score">
                    Confidence: {confidence * 100:.1f}%
                    <i class="{confidence_icon} ms-2"></i>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence * 100}%"></div>
                </div>
                
                <div class="probabilities">
                    <h6 class="mb-3"><i class="fas fa-chart-bar me-2"></i>All Category Probabilities</h6>
                    {prob_html}
                </div>
            </div>
        '''
        
        html_content = self.get_base_html_with_content(content, input_text)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-length', str(len(html_content)))
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def get_base_html_with_content(self, content, input_text=''):
        """Get base HTML with injected content."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Classifier - Result</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            position: relative;
            overflow-x: hidden;
        }}

        .main-container {{
            position: relative;
            z-index: 1;
            padding: 2rem 0;
        }}

        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: none;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        }}

        .card-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 20px 20px 0 0 !important;
            border: none;
            text-align: center;
            padding: 2rem;
        }}

        .card-header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .form-control {{
            border: 2px solid #e9ecef;
            border-radius: 15px;
            padding: 1rem;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.9);
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .btn-secondary {{
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            border: none;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: white;
        }}

        .alert {{
            border: none;
            border-radius: 15px;
            padding: 1.5rem;
            font-size: 1rem;
            margin-top: 1.5rem;
            animation: slideIn 0.5s ease-out;
        }}

        .alert-danger {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
            color: #721c24;
        }}

        .prediction-result {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }}

        .prediction-category {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}

        .confidence-score {{
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            opacity: 0.9;
        }}

        .confidence-bar {{
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            height: 10px;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }}

        .confidence-fill {{
            background: rgba(255, 255, 255, 0.8);
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease-out;
            animation: fillBar 1s ease-out;
        }}

        .probabilities {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
        }}

        .probability-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }}

        .probability-item:last-child {{
            margin-bottom: 0;
        }}

        .probability-bar {{
            width: 100px;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-left: 1rem;
        }}

        .probability-fill {{
            height: 100%;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 4px;
            transition: width 1s ease-out;
        }}

        .category-icons {{
            font-size: 3rem;
            margin-bottom: 1rem;
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fillBar {{
            from {{
                width: 0%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container main-container">
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="card">
                    <div class="card-header">
                        <h1><i class="fas fa-newspaper"></i> News Classifier</h1>
                    </div>
                    <div class="card-body p-4">
                        <form method="POST" action="/predict" id="newsForm">
                            <div class="mb-4">
                                <label for="news_text" class="form-label">
                                    <i class="fas fa-edit"></i> Enter News Article Text
                                </label>
                                <textarea 
                                    class="form-control" 
                                    id="news_text" 
                                    name="news_text" 
                                    rows="6" 
                                    placeholder="Paste your news article here... (minimum 10 characters)"
                                    required>{input_text}</textarea>
                                <div class="form-text">
                                    Categories: Politics, Sports, Business, Technology, Entertainment
                                </div>
                            </div>
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-magic"></i>
                                    Classify News
                                </button>
                                <a href="/" class="btn btn-secondary">
                                    <i class="fas fa-home"></i>
                                    Start Over
                                </a>
                            </div>
                        </form>

                        {content}
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''

def main():
    """Main function to start the server."""
    PORT = 8000
    
    print(f"Starting News Classifier Server on http://localhost:{PORT}")
    print("Available endpoints:")
    print("  GET  /          - Main page")
    print("  POST /predict   - Form-based prediction")
    print("  POST /api/predict - JSON API prediction")
    print("  GET  /health    - Health check")
    print()
    
    with socketserver.TCPServer(("", PORT), NewsClassifierHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()