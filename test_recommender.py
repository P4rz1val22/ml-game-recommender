import sys

sys.path.append('')

from src.utils.data_loader import SteamJSONLoader
from src.models.content_based import ContentBasedRecommender
from pathlib import Path


def analyze_ownership_distribution(games_df):
    print("📊 Ownership Distribution Analysis:")
    print("=" * 50)

    ownership_counts = games_df['estimated_owners'].value_counts()
    print(f"Total games: {len(games_df):,}")
    print("\nTop ownership ranges:")
    for range_name, count in ownership_counts.head(10).items():
        percentage = (count / len(games_df)) * 100
        print(f"{range_name:20s}: {count:6,} ({percentage:5.1f}%)")

    # Check our filters
    zero_to_zero = len(games_df[games_df['estimated_owners'] == '0 - 0'])
    zero_to_20k = len(games_df[games_df['estimated_owners'] == '0 - 20000'])
    both_excluded = len(games_df[games_df['estimated_owners'].isin(['0 - 20000', '0 - 0'])])
    popular_only = len(games_df[~games_df['estimated_owners'].isin(['0 - 20000', '0 - 0'])])

    print(f"\n🔍 Filter Analysis:")
    print(f"'0 - 0' games:        {zero_to_zero:6,}")
    print(f"'0 - 20000' games:    {zero_to_20k:6,}")
    print(f"Both excluded:        {both_excluded:6,}")
    print(f"Remaining (popular):  {popular_only:6,}")


def test_recommender():
    print("🧪 Testing Content-Based Recommender...")

    # Load data
    current_file = Path(__file__)
    project_root = current_file.parent
    data_path = project_root / "data" / "steam_games.json"

    loader = SteamJSONLoader(str(data_path))
    games_df = loader.json_to_dataframe()
    analyze_ownership_distribution(games_df)

    # Initialize and fit recommender
    recommender = ContentBasedRecommender()
    recommender.fit_popular_games(games_df)

    # Test with some popular games (you can change these)
    test_games = ["292030", "730", "413150"]  # Witcher 3 (RPG) + Counter-Strike (FPS) + Stardew Valley (Casual)
    print(f"\n🎮 Testing with games: {test_games}")

    # Get recommendations
    recommendations = recommender.recommend_popular(test_games, n=5)

    print(f"\n🎯 Got {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Similarity: {rec['similarity_score']:.3f}")
        print(f"   Genres: {rec['genres']}")
        print(f"   Reviews: {rec['positive_reviews']} positive")
        print()


if __name__ == "__main__":
    test_recommender()
