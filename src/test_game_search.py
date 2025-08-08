# test_game_search.py
from utils.data_loader import RAWGDatasetLoader
from models.content_based import ContentBasedRecommender
from services.game_search import GameSearchService

# Load data and fit recommender
loader = RAWGDatasetLoader()
df = loader.load_and_process_data()
popular_df = loader.get_engagement_based_popular_games(debug=False)

recommender = ContentBasedRecommender()
recommender.fit_popular_games(popular_df)

# Initialize search service
search_service = GameSearchService(recommender.popular_games, recommender.game_id_to_index)

# Test searches
print("🔍 Testing fuzzy search:")
results = search_service.search_games_by_name("Half-Life")
for result in results[:3]:
    print(f"- {result['name']} (Score: {result['match_score']}, ID: {result['game_id']})")

results = search_service.search_games_by_name("half life")
for result in results[:3]:
    print(f"- {result['name']} (Score: {result['match_score']}, ID: {result['game_id']})")

results = search_service.search_games_by_name("god o war")
for result in results[:3]:
    print(f"- {result['name']} (Score: {result['match_score']}, ID: {result['game_id']})")

results = search_service.search_games_by_name("minekraft")
for result in results[:3]:
    print(f"- {result['name']} (Score: {result['match_score']}, ID: {result['game_id']})")

# Test ID lookup
print(f"\n🎯 Testing ID lookup:")
game_info = search_service.get_game_by_id("19103")
if game_info:
    print(f"Found: {game_info['name']} ({game_info['release_date']})")
