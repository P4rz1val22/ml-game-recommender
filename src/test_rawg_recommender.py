# test_rawg_recommender.py
from utils.data_loader import RAWGDatasetLoader
from models.content_based import ContentBasedRecommender

# Load RAWG data
loader = RAWGDatasetLoader()
df = loader.load_and_process_data()
popular_df = loader.get_engagement_based_popular_games()

# Initialize and fit recommender
recommender = ContentBasedRecommender()
recommender.fit_popular_games(popular_df)

# Test with a few game IDs from your 2025 list
test_games = ["28"]  # Use actual IDs from your data
recommendations = recommender.recommend_popular(test_games, n=5)

print("🎯 Test Recommendations:")
for rec in recommendations:
    print(f"- {rec['name']} (similarity: {rec['similarity_score']:.3f})")
