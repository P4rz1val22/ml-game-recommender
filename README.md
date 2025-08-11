# 🎮 RAWG Hybrid ML Game Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![ML](https://img.shields.io/badge/ML-Ensemble_Models-orange.svg)](https://scikit-learn.org)
[![Data](https://img.shields.io/badge/Dataset-RAWG_878K_Games-purple.svg)](https://huggingface.co/datasets/atalaydenknalbant/rawg-games-dataset)

## 🚀 Overview

A machine learning recommendation system that combines **semantic understanding**, **genre
classification**, and **tag-based feature matching** to provide personalized game recommendations.
Built with enterprise-grade architecture and designed for integration with conversational AI
systems.

### **Key Features**

- 🧠 **Hybrid ML Models**: 4 recommendation algorithms with configurable weights
- 🔍 **Advanced Filtering**: Franchise, platform, date, and quality filters
- 🌍 **Multi-Platform**: Covers PC, PlayStation, Xbox, Nintendo, Mobile gaming
- ⚡ **High Performance**: Sub-1.5s response times with comprehensive caching
- 📊 **Transparent AI**: Model breakdown and explanation generation

---

## 🛠️ Technical Stack

### **Core Technologies**

- **FastAPI**: Modern async web framework with automatic documentation
- **Sentence Transformers**: Semantic similarity using pre-trained language models
- **scikit-learn**: TF-IDF vectorization and cosine similarity calculations
- **pandas/numpy**: High-performance data processing and linear algebra
- **RAWG Database**: 878K games across all major gaming platforms

### **ML Architecture**

```python
# Ensemble recommendation system
final_score = (
        0.6 * semantic_similarity +  # Description embeddings
        0.3 * genre_rarity_score +  # Weighted genre matching  
        0.1 * tag_tfidf_score  # Feature-based similarity
)
```

---

## 📊 Dataset & Performance

### **RAWG Video Games Database**

- **Scale**: 878,539 games with quality descriptions
- **Coverage**: Multi-platform (PC, PlayStation, Xbox, Nintendo, Mobile, Web)
- **Features**: Rich metadata (genres, tags, ratings, platforms, release dates)
- **License**: CC0 (Commercial use allowed)
- **Engagement Filtering**: 11,335 popular games for recommendations

### **Performance Benchmarks**

| Component        | Response Time | Memory Usage | Cache Benefits       |
|------------------|---------------|--------------|----------------------|
| Data Loading     | <1s (cached)  | 200MB        | 99.7% reduction      |
| Content Model    | 33ms          | 150MB        | Permanent embeddings |
| Genre Model      | 5ms           | 10MB         | Instant vector ops   |
| Tag Model        | 55ms          | 50MB         | TF-IDF matrix cached |
| Hybrid + Filters | 300-1500ms    | 250MB        | Advanced filtering   |

---

## 🚀 Quick Start

### **Installation**

```bash
git clone https://github.com/your-username/ml-game-recommender
cd ml-game-recommender
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Run API Server**

```bash
python src/main.py
```

**API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)

### **Basic Usage**

```bash
# Get recommendations for Left 4 Dead 2
curl -X POST "http://localhost:8000/api/recommendations/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"liked_games": ["12020"], "limit": 5}'

# Search for games
curl "http://localhost:8000/api/games/search?name=half%20life&limit=5"

# Get game details
curl "http://localhost:8000/api/games/19103"
```

---

## 🤖 API Endpoints

### **Recommendation Endpoints**

```http
POST /api/recommendations/description
POST /api/recommendations/genre  
POST /api/recommendations/tags
POST /api/recommendations/hybrid
```

**Request Format:**

```json
{
  "liked_games": [
    "19103",
    "12020"
  ],
  "limit": 10,
  "exclude_owned": true,
  "weights": {
    "description": 0.6,
    "genre": 0.3,
    "tags": 0.1
  },
  "filters": {
    "exclude_franchise": true,
    "platforms": [
      "PC",
      "PlayStation"
    ],
    "min_year": 2020,
    "min_rating": 3.5,
    "min_reviews": 100
  }
}
```

### **Search Endpoints**

```http
GET /api/games/search?name={query}&limit={limit}
GET /api/games/{game_id}
```

### **System Endpoints**

```http
GET /api/recommendations/models/status
GET /api/recommendations/models/analysis
GET /health
```

---

## 🎯 Advanced Features

### **1. Configurable Model Weights**

Customize recommendation strategy by adjusting model importance:

```
// Balanced (default) - Best overall results
{
  "description": 0.6,
  "genre": 0.3,
  "tags": 0.1
}

// Thematic explorer - Cross-genre theme discovery
{
  "description": 0.8,
  "genre": 0.1,
  "tags": 0.1
}

// Genre loyalist - Consistent category matching
{
  "description": 0.2,
  "genre": 0.6,
  "tags": 0.2
}

// Mechanic focused - Specific feature matching
{
  "description": 0.2,
  "genre": 0.2,
  "tags": 0.6
}
```

### **2. Advanced Filtering Options**

**Franchise Filtering:**

```
{
  "exclude_franchise": true
}  // No Left 4 Dead 1 when you like Left 4 Dead 2
```

**Platform Filtering:**

```
{
  "platforms": [
    "PC",
    "PlayStation"
  ]
}  // Only games on specified platforms
```

**Date Range Filtering:**

```
{
  "min_year": 2020
}          // Only modern games
{
  "max_year": 2015
}          // Only retro games  
{
  "min_year": 2018,
  "max_year": 2023
}  // Specific time window
```

**Quality Filtering:**

```
{
  "min_rating": 4.0,
  "min_reviews": 500
}  // Only highly-rated, well-reviewed games
```

### **3. Response Format with Model Transparency**

```json
{
  "recommendations": [
    {
      "game_id": "13537",
      "name": "Half-Life 2",
      "similarity_score": 0.664,
      "reason": "Hybrid match: Semantic themes + genre/tag support (franchise filtered)",
      "model_breakdown": {
        "description": 0.518,
        "genre": 0.973,
        "tags": 0.606
      }
    }
  ],
  "algorithm": "hybrid_d0.6_g0.3_t0.1_franchise_filtered",
  "processing_time_ms": 314.8
}
```

---

## 🧠 Machine Learning Details

### **ContentBasedRecommender**

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Input**: Game description text
- **Output**: 384-dimensional semantic embeddings
- **Similarity**: Cosine similarity between averaged user preferences
- **Strength**: Discovers thematic connections across different genres

### **GenreBasedRecommender**

- **Algorithm**: Rarity-weighted genre vector similarity
- **Input**: Game genre classifications (19 unique genres)
- **Processing**: Logarithmic rarity weighting (rare genres weighted higher)
- **Similarity**: Cosine similarity between weighted genre vectors
- **Strength**: Perfect category filtering with intelligent rarity bias

### **TagBasedRecommender**

- **Algorithm**: TF-IDF with platform tag filtering
- **Input**: Pipe-separated gameplay tags (1000+ unique tags)
- **Processing**: Automatic rarity weighting via TF-IDF, platform tag removal
- **Similarity**: Cosine similarity between TF-IDF vectors
- **Strength**: Specific mechanic and feature matching

### **HybridRecommender**

- **Algorithm**: Weighted ensemble with advanced filtering
- **Input**: Combined similarity scores from all three models
- **Processing**: Configurable weighted averaging + comprehensive filtering
- **Output**: Best recommendations with model transparency
- **Strength**: Combines all approaches with user customization

---

## 🔧 Development Setup

### **Project Structure**

```
ml-game-recommender/
├── src/
│   ├── api/
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── recommendations.py     # ML recommendation endpoints
│   │   └── games.py              # Game search endpoints
│   ├── models/
│   │   ├── content_based.py      # Semantic similarity model
│   │   ├── genre_based.py        # Genre rarity model
│   │   ├── tag_based.py          # TF-IDF tag model
│   │   └── hybrid_recommender.py # Ensemble model + filtering
│   ├── services/
│   │   └── game_search.py        # Fuzzy search service
│   ├── utils/
│   │   └── data_loader.py        # RAWG dataset loader
│   └── main.py                   # FastAPI application
├── cache/                        # Model and data caching
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### **Environment Setup**

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn sentence-transformers datasets

# Set environment variables (optional)
export PYTHONPATH="${PYTHONPATH}:src"

# Run development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing

### **Postman Collection**

Import the complete test suite from `test.postman_collection.json`:

- **39 test requests** covering all endpoints
- **Error case validation** for robustness testing
- **Performance benchmarks** for load testing
- **Filter combinations** for advanced feature validation

### **Unit Testing**

```bash
# Test individual models
python src/models/content_based.py
python src/models/genre_based.py  
python src/models/tag_based.py
python src/models/hybrid_recommender.py
```

### **Integration Testing**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/recommendations/models/status
```

---

## 📊 Technical Specifications

### **System Requirements**

- **Python**: 3.12+
- **Memory**: 512MB minimum, 1GB recommended
- **Storage**: 500MB for cached models and embeddings
- **CPU**: Multi-core recommended for sentence transformer inference

### **Dependencies**

```
fastapi>=0.104.0
uvicorn>=0.20.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
datasets>=2.14.0
pydantic>=2.0.0
```

### **API Limits**

- **Request Rate**: No limits (development)
- **Response Size**: 50 recommendations maximum per request
- **Search Results**: 25 games maximum per search
- **Concurrent Users**: Thread-safe, supports multiple simultaneous requests

---

## 🏗️ Architecture Decisions

### **Why Hybrid Ensemble?**

- **Industry Standard**: Used by Netflix, Spotify, Amazon for production recommendations
- **Complementary Strengths**: Each model captures different similarity aspects
- **User Flexibility**: Configurable weights adapt to different preference types
- **Quality Assurance**: Multiple models reduce individual model biases

### **Why RAWG Database?**

- **Scale**: 8x larger than Steam-only datasets (878K vs 111K games)
- **Coverage**: Multi-platform support beyond PC gaming
- **Quality**: Professional metadata with rich descriptions and engagement metrics
- **Legal**: CC0 license allows commercial and portfolio use

### **Why Advanced Filtering?**

- **User Experience**: Prevents recommendation fatigue from obvious suggestions
- **Discovery**: Encourages exploration of new franchises and genres
- **Personalization**: Platform-specific and quality-based customization
- **Chatbot Integration**: Natural language filtering through conversational context

---

## 🔗 Integration Examples

### **Basic Recommendation Flow**

```python
import requests

# 1. Search for a game
search_result = requests.get("http://localhost:8000/api/games/search",
                             params={"name": "cyberpunk", "limit": 3})

# 2. Get game ID
game_id = search_result.json()['games'][0]['game_id']

# 3. Get recommendations  
recommendations = requests.post("http://localhost:8000/api/recommendations/hybrid",
                                json={"liked_games": [game_id], "limit": 5})

# 4. Process results
for game in recommendations.json()['recommendations']:
    print(f"Recommended: {game['name']} (Score: {game['similarity_score']:.3f})")
```

### **Advanced Filtering Example**

```python
# Modern high-quality PC games, no franchise repetition
response = requests.post("http://localhost:8000/api/recommendations/hybrid", json={
    "liked_games": ["12020"],  # Left 4 Dead 2
    "limit": 10,
    "weights": {"description": 0.7, "genre": 0.2, "tags": 0.1},
    "filters": {
        "exclude_franchise": True,
        "platforms": ["PC"],
        "min_year": 2020,
        "min_rating": 4.0,
        "min_reviews": 200
    }
})
```

### **Chatbot Integration Pattern**

```python
class GameRecommendationBot:
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_url = api_base_url

    def search_games(self, user_input: str):
        """Handle natural language game search"""
        return requests.get(f"{self.api_url}/api/games/search",
                            params={"name": user_input, "limit": 5})

    def get_recommendations(self, game_ids: List[str], user_preferences: dict):
        """Get personalized recommendations"""
        return requests.post(f"{self.api_url}/api/recommendations/hybrid",
                             json={
                                 "liked_games": game_ids,
                                 "weights": user_preferences.get("weights"),
                                 "filters": user_preferences.get("filters", {})
                             })

    def explain_recommendation(self, recommendation: dict):
        """Generate natural language explanation"""
        breakdown = recommendation['model_breakdown']
        reason = recommendation['reason']
        return f"I recommend {recommendation['name']} because {reason}. "
        f"It matches your preferences with {breakdown['description']:.2f} thematic similarity."
```

---

## 📋 API Reference

### **Recommendation Request**

```json
{
  "liked_games": [
    "game_id_1",
    "game_id_2"
  ],
  "limit": 10,
  "exclude_owned": true,
  "weights": {
    "description": 0.6,
    "genre": 0.3,
    "tags": 0.1
  },
  "filters": {
    "exclude_franchise": false,
    "platforms": [
      "PC",
      "PlayStation"
    ],
    "min_year": 2020,
    "max_year": 2024,
    "min_rating": 3.5,
    "min_reviews": 100
  }
}
```

### **Recommendation Response**

```json
{
  "recommendations": [
    {
      "game_id": "13537",
      "name": "Half-Life 2",
      "description": "Gordon Freeman became the most popular...",
      "genres": "Action|Shooter",
      "tags": "Singleplayer|Atmospheric|Story Rich|...",
      "rating": 4.48,
      "similarity_score": 0.664,
      "reason": "Hybrid match: Semantic themes + genre/tag support",
      "model_breakdown": {
        "description": 0.518,
        "genre": 0.973,
        "tags": 0.606
      }
    }
  ],
  "total_found": 5,
  "algorithm": "hybrid_d0.6_g0.3_t0.1_franchise_filtered",
  "processing_time_ms": 314.8
}
```

---

## 🔬 Testing & Validation

### **Model Validation**

- **Genre Model**: Perfect category consistency (Action|Shooter → Action|Shooter games)
- **Content Model**: Cross-genre thematic discovery (zombie themes across genres)
- **Tag Model**: Specific mechanic matching (co-op, atmospheric, story-rich)
- **Hybrid Model**: Weighted combination producing diverse, relevant results

### **Filter Validation**

- **Franchise**: Successfully excludes same-series games
- **Platform**: Accurate multi-platform matching
- **Date**: Precise year-based filtering
- **Quality**: Effective rating and review thresholds

### **Performance Validation**

- **Response Time**: All endpoints under 1.5s target
- **Memory Usage**: Stable 250MB total footprint
- **Error Handling**: Comprehensive edge case coverage
- **Concurrent Load**: Thread-safe operation verified

---

## 📞 Support & Documentation

### **Interactive Documentation**

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### **Testing Resources**

- **Postman Collection**: Complete test suite with 39+ requests

---

## 🏆 Project Achievements

**This ML recommendation system demonstrates:**

### **Advanced ML Engineering:**

- Multi-modal ensemble methods used by industry leaders
- Feature engineering across text, categorical, and tag data
- Production-ready caching and performance optimization
- Model interpretability and transparent decision-making

### **Full-Stack Development:**

- Modern async API with comprehensive documentation
- Sophisticated filtering and configuration systems
- Professional error handling and validation
- Scalable architecture supporting microservice deployment

### **Product Engineering:**

- User-centric design solving real recommendation problems
- Flexible configuration for different use cases and preferences
- Quality assurance through multi-dimensional filtering
- Integration-ready design for conversational AI systems

**Built in 1 week as part of an 8-week coding portfolio journey. Ready for Week 6 AI chatbot
integration.** 🚀

---

## 📄 License

MIT License - Feel free to use this code for learning, portfolio projects, or commercial
applications.

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome! Feel free to open issues or
submit pull requests.

---

**Made with ❤️ by Luis Sarmiento**