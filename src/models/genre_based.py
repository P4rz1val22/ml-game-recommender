import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GenreBasedRecommender:
    def __init__(self, games_df: pd.DataFrame, cache_dir: str = 'cache'):
        self.games_df = games_df
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache paths
        self.genre_weights_cache = self.cache_dir / "genre_weights.pkl"
        self.genre_vectors_cache = self.cache_dir / "genre_vectors.npy"
        self.genre_mapping_cache = self.cache_dir / "genre_mapping.pkl"

        # Core data structures
        self.genre_weights: Dict[str, float] = {}
        self.game_genre_vectors: Optional[np.ndarray] = None
        self.game_id_to_index: Dict[str, int] = {}
        self.all_genres: Set[str] = set()

        logger.info(f"🎨 Initializing GenreBasedRecommender with {len(games_df):,} games")

        # Initialize the recommender
        self._initialize()

    def _initialize(self):
        """Initialize the recommender with caching"""
        if self._load_from_cache():
            logger.info("🚀 Loaded genre model from cache!")
        else:
            logger.info("🔨 Building genre model from scratch...")
            self._build_model()
            self._save_to_cache()

    def _load_from_cache(self) -> bool:
        """Try to load from cache"""
        try:
            if (self.genre_weights_cache.exists() and
                    self.genre_vectors_cache.exists() and
                    self.genre_mapping_cache.exists()):
                import pickle

                with open(self.genre_weights_cache, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.genre_weights = cache_data['genre_weights']
                    self.all_genres = cache_data['all_genres']

                self.game_genre_vectors = np.load(self.genre_vectors_cache)

                with open(self.genre_mapping_cache, 'rb') as f:
                    self.game_id_to_index = pickle.load(f)

                return True
        except Exception as e:
            logger.warning(f"⚠️ Genre cache loading failed: {e}")
            return False

        return False

    def _save_to_cache(self):
        """Save to cache"""
        try:
            import pickle

            cache_data = {
                'genre_weights': self.genre_weights,
                'all_genres': self.all_genres
            }

            with open(self.genre_weights_cache, 'wb') as f:
                pickle.dump(cache_data, f)

            np.save(self.genre_vectors_cache, self.game_genre_vectors)

            with open(self.genre_mapping_cache, 'wb') as f:
                pickle.dump(self.game_id_to_index, f)

            logger.info("✅ Genre model cached successfully!")

        except Exception as e:
            logger.error(f"❌ Genre cache saving failed: {e}")

    def _build_model(self):
        """Build the genre-based model"""
        # Step 1: Calculate genre rarity weights
        self._calculate_genre_rarity_weights()

        # Step 2: Create genre vectors for all games
        self._create_genre_vectors()

        # Step 3: Create game ID mapping
        self._create_id_mapping()

    def _calculate_genre_rarity_weights(self):
        """Calculate rarity weights for genres - rare genres get higher weights"""
        logger.info("📊 Calculating genre rarity weights...")

        genre_counts = {}
        total_games_with_genres = 0

        # Count all genre occurrences
        for genres_str in self.games_df['genres'].dropna():
            if genres_str and str(genres_str).strip():  # Skip empty strings
                total_games_with_genres += 1

                # Split by | if multiple genres, otherwise single genre
                if '|' in str(genres_str):
                    genres = [g.strip() for g in str(genres_str).split('|')]
                else:
                    genres = [str(genres_str).strip()]

                for genre in genres:
                    if genre:  # Skip empty strings
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        self.all_genres.add(genre)

        # Calculate rarity weights: log(total / frequency)
        # Rare genres get higher weights
        for genre, count in genre_counts.items():
            self.genre_weights[genre] = np.log(total_games_with_genres / count)

        # Sort genres by rarity for debugging
        sorted_genres = sorted(self.genre_weights.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"✅ Calculated weights for {len(self.genre_weights)} unique genres")
        logger.info(f"📈 Rarest genres (highest weights):")
        for genre, weight in sorted_genres[:5]:
            count = genre_counts[genre]
            logger.info(f"   {genre}: {weight:.3f} (appears in {count:,} games)")

        logger.info(f"📉 Most common genres (lowest weights):")
        for genre, weight in sorted_genres[-5:]:
            count = genre_counts[genre]
            logger.info(f"   {genre}: {weight:.3f} (appears in {count:,} games)")

    def _create_genre_vectors(self):
        """Create weighted genre vectors for all games"""
        logger.info("🔢 Creating weighted genre vectors...")

        # Create matrix: [num_games, num_genres]
        num_games = len(self.games_df)
        num_genres = len(self.all_genres)

        # Map genre names to indices
        genre_to_index = {genre: idx for idx, genre in enumerate(sorted(self.all_genres))}

        # Initialize genre matrix
        self.game_genre_vectors = np.zeros((num_games, num_genres))

        # Fill genre vectors
        for game_idx, genres_str in enumerate(self.games_df['genres']):
            if pd.notna(genres_str) and str(genres_str).strip():
                # Parse genres
                if '|' in str(genres_str):
                    genres = [g.strip() for g in str(genres_str).split('|')]
                else:
                    genres = [str(genres_str).strip()]

                # Set weighted values for each genre
                for genre in genres:
                    if genre in genre_to_index and genre in self.genre_weights:
                        genre_idx = genre_to_index[genre]
                        # Use rarity weight as the vector value
                        self.game_genre_vectors[game_idx, genre_idx] = self.genre_weights[genre]

        logger.info(f"✅ Created genre vectors: {self.game_genre_vectors.shape}")
        logger.info(f"📐 Non-zero entries: {np.count_nonzero(self.game_genre_vectors):,}")

    def _create_id_mapping(self):
        """Create game ID to index mapping"""
        self.game_id_to_index = {
            str(game_id): idx for idx, game_id in enumerate(self.games_df['id'])
        }
        logger.info(f"🗺️ Created game ID mapping for {len(self.game_id_to_index):,} games")

    def _get_user_genre_preference(self, liked_game_ids: List[str]) -> Optional[np.ndarray]:
        """Calculate user's genre preference vector from liked games"""
        valid_vectors = []
        found_games = []

        for game_id in liked_game_ids:
            if str(game_id) in self.game_id_to_index:
                idx = self.game_id_to_index[str(game_id)]
                game_vector = self.game_genre_vectors[idx]

                # Only include if game has genres (non-zero vector)
                if np.any(game_vector > 0):
                    valid_vectors.append(game_vector)
                    found_games.append(self.games_df.iloc[idx]['name'])

        if not valid_vectors:
            logger.warning(f"❌ No valid games with genres found from: {liked_game_ids}")
            return None

        logger.info(f"✅ Found {len(valid_vectors)} games with genres: {found_games}")

        # Average the genre vectors to create user preference
        user_preference = np.mean(valid_vectors, axis=0)
        return user_preference

    def get_similarities(self, liked_game_ids: List[str]) -> np.ndarray:
        """Get genre similarity scores for all games (for hybrid use)"""
        user_preference = self._get_user_genre_preference(liked_game_ids)
        if user_preference is None:
            return np.zeros(len(self.games_df))

        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([user_preference], self.game_genre_vectors)[0]

        return similarities

    def recommend(
            self,
            liked_game_ids: List[str],
            n: int = 10,
            exclude_owned: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate genre-based recommendations"""

        # Get similarity scores
        similarities = self.get_similarities(liked_game_ids)

        if np.all(similarities == 0):
            logger.warning("No genre similarities found")
            return []

        # Exclude owned games if requested
        if exclude_owned:
            owned_indices = [
                self.game_id_to_index[str(game_id)]
                for game_id in liked_game_ids
                if str(game_id) in self.game_id_to_index
            ]
            similarities[owned_indices] = -1

        # Get top N recommendations
        top_indices = np.argsort(similarities)[::-1][:n * 2]  # Get extra to filter out low scores

        recommendations = []
        for idx in top_indices:
            if similarities[idx] <= 0:  # Skip excluded or zero similarity
                continue

            game = self.games_df.iloc[idx]

            # Get the genre overlap explanation
            genre_explanation = self._get_genre_explanation(
                liked_game_ids, str(game['id'])
            )

            recommendations.append({
                'game_id': str(game['id']),
                'name': game['name'],
                'description': str(game['description_raw'])[:200] + '...' if pd.notna(game['description_raw']) else 'No description',
                'release_date': str(game['released'])[:10] if pd.notna(game['released']) else 'Unknown',
                'genres': str(game['genres']) if pd.notna(game['genres']) else '',
                'tags': str(game['tags'])[:100] + '...' if pd.notna(game['tags']) and len(str(game['tags'])) > 100 else str(game['tags']) if pd.notna(game['tags']) else '',
                'rating': float(game['rating']) if pd.notna(game['rating']) else 0.0,
                'ratings_count': int(game['ratings_count']) if pd.notna(game['ratings_count']) else 0,
                'added_to_library': int(game['added']) if pd.notna(game['added']) else 0,
                'platforms': str(game['platforms']) if pd.notna(game['platforms']) else '',
                'similarity_score': float(similarities[idx]),
                'reason': f"Genre match: {genre_explanation}"
            })

            if len(recommendations) >= n:
                break

        logger.info(f"🎯 Generated {len(recommendations)} genre-based recommendations")
        return recommendations

    def _get_genre_explanation(self, liked_game_ids: List[str], recommended_game_id: str) -> str:
        """Create explanation for why this game was recommended based on genres"""

        # Get genres from liked games
        liked_genres = set()
        for game_id in liked_game_ids:
            if str(game_id) in self.game_id_to_index:
                idx = self.game_id_to_index[str(game_id)]
                genres_str = self.games_df.iloc[idx]['genres']
                if pd.notna(genres_str) and str(genres_str).strip():
                    if '|' in str(genres_str):
                        genres = [g.strip() for g in str(genres_str).split('|')]
                    else:
                        genres = [str(genres_str).strip()]
                    liked_genres.update(genres)

        # Get genres from recommended game
        if str(recommended_game_id) in self.game_id_to_index:
            idx = self.game_id_to_index[str(recommended_game_id)]
            rec_genres_str = self.games_df.iloc[idx]['genres']

            if pd.notna(rec_genres_str) and str(rec_genres_str).strip():
                if '|' in str(rec_genres_str):
                    rec_genres = [g.strip() for g in str(rec_genres_str).split('|')]
                else:
                    rec_genres = [str(rec_genres_str).strip()]

                # Find common genres
                common_genres = liked_genres.intersection(rec_genres)

                if common_genres:
                    # Sort by rarity (highest weight first)
                    sorted_common = sorted(
                        common_genres,
                        key=lambda g: self.genre_weights.get(g, 0),
                        reverse=True
                    )
                    return f"Shares {', '.join(sorted_common[:2])}"
                else:
                    return f"Similar genre profile"

        return "Genre similarity"

    def get_game_info(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific game"""
        if str(game_id) not in self.game_id_to_index:
            return None

        idx = self.game_id_to_index[str(game_id)]
        game = self.games_df.iloc[idx]

        return {
            'game_id': str(game['id']),
            'name': game['name'],
            'description': str(game['description_raw'])[:300] + '...' if pd.notna(game['description_raw']) else 'No description',
            'genres': str(game['genres']) if pd.notna(game['genres']) else '',
            'rating': float(game['rating']) if pd.notna(game['rating']) else 0.0,
            'ratings_count': int(game['ratings_count']) if pd.notna(game['ratings_count']) else 0,
            'added_to_library': int(game['added']) if pd.notna(game['added']) else 0,
            'platforms': str(game['platforms']) if pd.notna(game['platforms']) else ''
        }

    def get_genre_analysis(self) -> Dict[str, Any]:
        """Get analysis of genre distribution and weights"""

        # Top rarest genres (highest weights)
        sorted_genres = sorted(self.genre_weights.items(), key=lambda x: x[1], reverse=True)

        # Count total games per genre for context
        genre_game_counts = {}
        for genres_str in self.games_df['genres'].dropna():
            if str(genres_str).strip():
                if '|' in str(genres_str):
                    genres = [g.strip() for g in str(genres_str).split('|')]
                else:
                    genres = [str(genres_str).strip()]

                for genre in genres:
                    if genre:
                        genre_game_counts[genre] = genre_game_counts.get(genre, 0) + 1

        return {
            'total_genres': len(self.all_genres),
            'rarest_genres': [
                {
                    'genre': genre,
                    'rarity_weight': round(weight, 3),
                    'game_count': genre_game_counts.get(genre, 0)
                }
                for genre, weight in sorted_genres[:10]
            ],
            'most_common_genres': [
                {
                    'genre': genre,
                    'rarity_weight': round(weight, 3),
                    'game_count': genre_game_counts.get(genre, 0)
                }
                for genre, weight in sorted_genres[-10:]
            ],
            'games_with_genres': int(np.sum(np.any(self.game_genre_vectors > 0, axis=1))),
            'games_without_genres': len(self.games_df) - int(np.sum(np.any(self.game_genre_vectors > 0, axis=1)))
        }


# Testing function
if __name__ == "__main__":
    # Add src to path for imports
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from utils.data_loader import RAWGDatasetLoader

    print("🎮 Testing GenreBasedRecommender")
    print("=" * 50)

    # Load data
    loader = RAWGDatasetLoader()
    df = loader.load_and_process_data()

    # Get popular games for testing
    popular_df = loader.get_engagement_based_popular_games(debug=False)

    print(f"📊 Testing on {len(popular_df):,} popular games")

    # Initialize recommender
    recommender = GenreBasedRecommender(popular_df)

    # Test recommendations
    test_games = ["19103", "12020"]  # Half-Life 2, Left 4 Dead 2
    print(f"\n🧪 Testing with games: {test_games}")

    recommendations = recommender.recommend(test_games, n=5)

    print(f"\n🎯 Top 5 Genre-Based Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Genres: {rec['genres']}")
        print(f"   Similarity: {rec['similarity_score']:.3f}")
        print(f"   Reason: {rec['reason']}")
        print()

    # Show genre analysis
    analysis = recommender.get_genre_analysis()
    print(f"\n📈 Genre Analysis:")
    print(f"Total unique genres: {analysis['total_genres']}")
    print(f"Games with genres: {analysis['games_with_genres']:,}")
    print(f"Games without genres: {analysis['games_without_genres']:,}")
