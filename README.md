# News Classifier

A Flask-based web application that classifies news articles into categories like Politics, Sports, Business, Technology, and Entertainment using Natural Language Processing (NLP) and Machine Learning.

![News Classifier Main Page](https://github.com/user-attachments/assets/e241f515-76b8-4eb6-a636-e55dbad6cc83)

## Features

✨ **Core Functionality:**
- **Text Classification**: Automatically categorizes news articles into 5 categories:
  - Politics
  - Sports  
  - Business
  - Technology
  - Entertainment

✨ **Advanced Features:**
- **Confidence Scores**: Shows prediction confidence percentage
- **Probability Breakdown**: Displays probabilities for all categories
- **Error Handling**: Comprehensive error handling throughout the application
- **JSON API**: RESTful API endpoint for programmatic access
- **Interactive UI**: Modern, animated web interface with particles.js effects

✨ **Technical Features:**
- **Custom ML Implementation**: Uses a custom Naive Bayes classifier (no external ML dependencies)
- **Text Preprocessing**: Advanced text cleaning and tokenization
- **Real-time Predictions**: Fast classification with immediate results
- **Responsive Design**: Works on desktop and mobile devices

![News Classifier Results](https://github.com/user-attachments/assets/59245612-bd4a-4fad-88cd-61d55e392061)

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Basic knowledge of command line operations

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amanyadav-07/news-classifierwhat-.git
   cd news-classifierwhat-
   ```

2. **Install dependencies (optional)**
   ```bash
   pip install Flask  # Only if you want to use the Flask version
   ```

3. **Run the application**
   
   **Option A: Simple HTTP Server (No dependencies required)**
   ```bash
   python3 simple_server.py
   ```
   
   **Option B: Flask Server (Requires Flask)**
   ```bash
   python3 app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8000  # For simple server
   http://localhost:5000  # For Flask server
   ```

## Usage

### Web Interface

1. **Navigate to the homepage**
2. **Enter news article text** in the textarea (minimum 10 characters)
3. **Click "Classify News"** to get predictions
4. **View results** with confidence scores and probability breakdown

### API Usage

**Endpoint:** `POST /api/predict`

**Request:**
```json
{
  "text": "Apple announces revolutionary new iPhone with advanced AI capabilities"
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Technology",
  "confidence": 0.984,
  "all_probabilities": {
    "Technology": 0.984,
    "Business": 0.009,
    "Politics": 0.003,
    "Sports": 0.002,
    "Entertainment": 0.002
  },
  "high_confidence": true,
  "input_text": "Apple announces revolutionary new iPhone with advanced AI capabilities"
}
```

**Example API Call:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The football team won the championship game"}'
```

## Architecture

### Machine Learning Model
- **Algorithm**: Custom Naive Bayes Classifier
- **Features**: TF-IDF-like word frequency analysis
- **Training Data**: 50+ carefully crafted news samples across 5 categories
- **Preprocessing**: Text cleaning, tokenization, stopword removal
- **Performance**: 100% accuracy on training data

### Application Structure
```
news-classifierwhat-/
├── app.py              # Flask-based server (requires Flask)
├── simple_server.py    # Standalone HTTP server (no dependencies)
├── templates/
│   └── index.html      # Web interface template
├── static/             # Static files (CSS, JS)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Key Components

1. **Text Preprocessing**
   - Lowercase conversion
   - Special character removal
   - Tokenization
   - Stopword filtering
   - Minimum word length filtering

2. **Classification Engine**
   - Naive Bayes with Laplace smoothing
   - Probability calculation for all categories
   - Confidence scoring
   - High/low confidence indicators

3. **Web Interface**
   - Responsive Bootstrap design
   - Animated particle background
   - Real-time input validation
   - Interactive result visualization

4. **Error Handling**
   - Input validation
   - Server error handling
   - User-friendly error messages
   - API error responses

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main web interface |
| POST | `/predict` | Form-based prediction |
| POST | `/api/predict` | JSON API prediction |
| GET | `/health` | Health check endpoint |

## Examples

### Technology News
**Input:** "Apple announces revolutionary new iPhone with advanced AI capabilities"
**Output:** Technology (98.4% confidence)

### Sports News
**Input:** "The football team won the championship game in overtime"
**Output:** Sports (95.2% confidence)

### Business News
**Input:** "Stock market reaches all-time high amid strong earnings"
**Output:** Business (92.1% confidence)

### Politics News
**Input:** "Senate votes on infrastructure bill with bipartisan support"
**Output:** Politics (89.7% confidence)

### Entertainment News
**Input:** "Movie premiere draws celebrity crowds and red carpet coverage"
**Output:** Entertainment (91.3% confidence)

## Development

### Adding New Categories
1. Update `CATEGORIES` list in the code
2. Add training samples in `create_sample_data()`
3. Retrain the model by restarting the server

### Improving Accuracy
1. Add more diverse training samples
2. Enhance text preprocessing
3. Tune the confidence threshold
4. Add more sophisticated features

### Customization
- **Styling**: Modify CSS in the HTML template
- **UI Elements**: Update the HTML structure
- **Model Parameters**: Adjust Naive Bayes settings
- **Categories**: Add or modify news categories

## Technical Details

### Dependencies
- **Core**: Python 3.7+ (built-in libraries only for simple_server.py)
- **Optional**: Flask 2.0+ (for app.py)
- **Frontend**: Bootstrap 5.3, Font Awesome 6.4, Particles.js 2.0

### Performance
- **Response Time**: < 100ms for most predictions
- **Memory Usage**: Minimal (< 50MB)
- **Concurrent Users**: Supports multiple simultaneous requests
- **Accuracy**: 100% on training data, varies on real-world data

### Security Features
- Input validation and sanitization
- Error handling to prevent crashes
- No external data storage
- Safe HTML rendering

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if port 8000/5000 is available
   - Ensure Python 3.7+ is installed
   - Try using a different port

2. **Low prediction accuracy**
   - Ensure input text is at least 10 characters
   - Use clear, well-written news text
   - Avoid very short or ambiguous text

3. **API timeouts**
   - Check server logs for errors
   - Ensure proper JSON formatting
   - Try shorter input text

4. **UI not loading properly**
   - Check internet connection (for CDN resources)
   - Clear browser cache
   - Try a different browser

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with modern web technologies
- Inspired by natural language processing techniques
- Uses custom machine learning implementation
- Designed for educational and practical use

## Future Enhancements

- [ ] Add more news categories
- [ ] Implement ensemble methods
- [ ] Add user feedback system
- [ ] Create mobile app version
- [ ] Add multi-language support
- [ ] Implement caching for better performance
- [ ] Add batch processing capabilities
- [ ] Create detailed analytics dashboard