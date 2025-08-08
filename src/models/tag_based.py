import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

logger = logging.getLogger(__name__)


class TagBasedRecommender:
    def __init__(self, games_df: pd.DataFrame, cache_dir: str = 'cache'):
        # Setting the dataframe
        self.games_df = games_df
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache
        self.tfidf_matrix_cache = self.cache_dir / "tag_tfidf_matrix.npy"
        self.tfidf_vectorizer_cache = self.cache_dir / "tag_tfidf_vectorizer.pkl"
        self.tag_mapping_cache = self.cache_dir / "tag_mapping.pkl"

        # Data structures
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tag_tfidf_matrix: Optional[np.ndarray] = None
        self.game_id_to_index: Dict[str, int] = {}
        self.feature_names: List[str] = []

        logger.info(f"🏷️ Initializing TagBasedRecommender with {len(games_df):,} games")

        self._initialize()

    def _initialize(self):
        """Initialize the recommender with caching"""
        logger.info("Trying cache...")
        if self._load_from_cache():
            logger.info("🚀 Loaded tag model from cache!")
        else:
            logger.info("🔨 No cache present, building tag TF-IDF model from scratch...")
            self._build_model()
            self._save_to_cache()

    def _load_from_cache(self) -> bool:
        """Try to load from cache"""
        try:
            if (self.tfidf_matrix_cache.exists() and
                    self.tfidf_vectorizer_cache.exists() and
                    self.tag_mapping_cache.exists()):
                import pickle

                # Load TF-IDF matrix
                self.tag_tfidf_matrix = np.load(self.tfidf_matrix_cache)

                # Load vectorizer model
                with open(self.tfidf_vectorizer_cache, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                    self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()

                # Load game mapping
                with open(self.tag_mapping_cache, 'rb') as f:
                    self.game_id_to_index = pickle.load(f)

                return True
        except Exception as e:
            logger.warning(f"⚠️ Tag cache loading failed: {e}")
            return False

        return False

    def _save_to_cache(self):
        """Save to cache"""
        try:
            import pickle  # hehe

            np.save(self.tfidf_matrix_cache, self.tag_tfidf_matrix)

            with open(self.tfidf_vectorizer_cache, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)

            with open(self.tag_mapping_cache, 'wb') as f:
                pickle.dump(self.game_id_to_index, f)

            logger.info("✅ Tag model cached successfully!")

        except Exception as e:
            logger.error(f"❌ Tag cache saving failed: {e}")

    def _build_model(self):
        """Build the tag-based TF-IDF model"""
        tag_documents = self._create_tag_documents()
        self._create_tfidf_matrix(tag_documents)
        self._create_id_mapping()

    def _create_tag_documents(self) -> List[str]:
        """Convert game tags to text documents for TF-IDF with cleanup"""
        logger.info("📝 Converting tags to documents with cleanup...")

        # Platform/metadata tags to filter out (lots from steam, not v useful)
        platform_tags = {
            'steam', 'achievements', 'controller', 'cloud', 'trading', 'cards',
            'workshop', 'support', 'overlay', 'remote', 'play', 'together',
            'in-app', 'purchases', 'leaderboards', 'stats', 'guide', 'available'
        }

        tag_documents = []
        games_with_tags = 0
        filtered_tag_count = 0

        for tags_str in self.games_df['tags']:
            if pd.notna(tags_str) and str(tags_str).strip():
                # Split tags by pipe
                raw_tags = [t.strip() for t in str(tags_str).split('|')]

                cleaned_tags = []
                for tag in raw_tags:
                    if tag:
                        tag_lower = tag.lower()

                        # Filter out platform tags
                        if not any(platform_word in tag_lower for platform_word in platform_tags):
                            kebab_tag = tag_lower.replace(' ', '-').replace('_', '-')
                            cleaned_tags.append(kebab_tag)
                        else:
                            filtered_tag_count += 1

                # Join cleaned tags with spaces for TF-IDF, otherwise won't work
                if cleaned_tags:
                    tag_document = ' '.join(cleaned_tags)
                    tag_documents.append(tag_document)
                    games_with_tags += 1
                else:
                    tag_documents.append("")
            else:
                tag_documents.append("")

        logger.info(f"✅ Created {len(tag_documents):,} tag documents")
        logger.info(f"📊 Games with tags: {games_with_tags:,} ({games_with_tags / len(tag_documents) * 100:.1f}%)")
        logger.info(f"🗑️ Filtered out {filtered_tag_count:,} platform/metadata tags")

        return tag_documents

    def _create_tfidf_matrix(self, tag_documents: List[str]):
        """Create TF-IDF matrix from tag documents"""
        logger.info("🔢 Building TF-IDF matrix...")

        # Rules for what tags are valid
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Top 1000 most important tags
            min_df=3,  # min appearances (number)
            max_df=0.6,  # max appearances %
            lowercase=False,
            token_pattern=r'\b[\w-]+\b',  # Include hyphens in tokens (preserves "co-op")
            stop_words=None
        )

        self.tag_tfidf_matrix = self.tfidf_vectorizer.fit_transform(tag_documents)

        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()

        logger.info(f"✅ TF-IDF matrix created: {self.tag_tfidf_matrix.shape}")
        logger.info(f"📐 Features (unique tags): {len(self.feature_names)}")
        logger.info(f"🔍 Sample top tags: {self.feature_names[:10]}")

        # Convert to dense array for faster similarity calculations
        if self.tag_tfidf_matrix.shape[1] <= 1000:  # Only if manageable size
            self.tag_tfidf_matrix = self.tag_tfidf_matrix.toarray()
            logger.info("📦 Converted to dense matrix for speed")

    def _create_id_mapping(self):
        """Create game ID to index mapping"""
        self.game_id_to_index = {
            str(game_id): idx for idx, game_id in enumerate(self.games_df['id'])
        }
        logger.info(f"🗺️ Created game ID mapping for {len(self.game_id_to_index):,} games")

    def _get_user_tag_preference(self, liked_game_ids: List[str]) -> Optional[np.ndarray]:
        """Calculate user's tag preference vector from liked games"""
        valid_vectors = []
        found_games = []

        # Adjusting user prefs through liked games
        for game_id in liked_game_ids:
            if str(game_id) in self.game_id_to_index:
                idx = self.game_id_to_index[str(game_id)]

                if hasattr(self.tag_tfidf_matrix, 'toarray'):
                    game_vector = self.tag_tfidf_matrix[idx].toarray().flatten()
                else:
                    game_vector = self.tag_tfidf_matrix[idx]
                if np.any(game_vector > 0):
                    valid_vectors.append(game_vector)
                    found_games.append(self.games_df.iloc[idx]['name'])

        if not valid_vectors:
            logger.warning(f"❌ No valid games with tags found from: {liked_game_ids}")
            return None

        logger.info(f"✅ Found {len(valid_vectors)} games with tags: {found_games}")

        # Average the TF-IDF vectors to create user preference
        user_preference = np.mean(valid_vectors, axis=0)
        return user_preference

    def get_similarities(self, liked_game_ids: List[str]) -> np.ndarray:
        """Get tag similarity scores for all games (for hybrid use)"""
        user_preference = self._get_user_tag_preference(liked_game_ids)
        if user_preference is None:
            return np.zeros(len(self.games_df))

        # Cosine similarities
        similarities = cosine_similarity([user_preference], self.tag_tfidf_matrix)[0]

        return similarities

    def recommend(
            self,
            liked_game_ids: List[str],
            n: int = 10,
            exclude_owned: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate tag-based recommendations"""

        # Get similarity scores
        similarities = self.get_similarities(liked_game_ids)

        if np.all(similarities == 0):
            logger.warning("No tag similarities found")
            return []

        # Exclude owned games (if requested)
        if exclude_owned:
            owned_indices = [
                self.game_id_to_index[str(game_id)]
                for game_id in liked_game_ids
                if str(game_id) in self.game_id_to_index
            ]
            similarities[owned_indices] = -1

        # Get top N recommendations
        top_indices = np.argsort(similarities)[::-1][:n * 2]

        recommendations = []
        for idx in top_indices:
            if similarities[idx] <= 0:  # Skip excluded or zero similarity
                continue

            game = self.games_df.iloc[idx]

            # Get the tag overlap explanation
            tag_explanation = self._get_tag_explanation(
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
                'reason': f"Tag match: {tag_explanation}"
            })

            if len(recommendations) >= n:
                break

        logger.info(f"🎯 Generated {len(recommendations)} tag-based recommendations")
        return recommendations

    def _get_tag_explanation(self, liked_game_ids: List[str], recommended_game_id: str) -> str:
        """Create explanation for why this game was recommended based on tags"""

        # Get tags from liked games
        liked_tags = set()
        for game_id in liked_game_ids:
            if str(game_id) in self.game_id_to_index:
                idx = self.game_id_to_index[str(game_id)]
                tags_str = self.games_df.iloc[idx]['tags']
                if pd.notna(tags_str) and str(tags_str).strip():
                    tags = [t.strip() for t in str(tags_str).split('|')]
                    liked_tags.update(tags)

        # Get tags from recommended game
        if str(recommended_game_id) in self.game_id_to_index:
            idx = self.game_id_to_index[str(recommended_game_id)]
            rec_tags_str = self.games_df.iloc[idx]['tags']

            if pd.notna(rec_tags_str) and str(rec_tags_str).strip():
                rec_tags = [t.strip() for t in str(rec_tags_str).split('|')]

                # Find common tags
                common_tags = liked_tags.intersection(rec_tags)

                if common_tags:
                    # Get TF-IDF weights for common tags to show most important ones
                    tag_weights = []
                    for tag in common_tags:
                        if tag.lower() in self.feature_names:
                            tag_idx = self.feature_names.index(tag.lower())
                            # Get the TF-IDF score for this tag in the recommended game
                            if hasattr(self.tag_tfidf_matrix, 'toarray'):
                                weight = self.tag_tfidf_matrix[idx, tag_idx]
                            else:
                                weight = self.tag_tfidf_matrix[idx][tag_idx]
                            tag_weights.append((tag, weight))

                    # Sort by TF-IDF weight (most important first)
                    tag_weights.sort(key=lambda x: x[1], reverse=True)

                    # Show top 2-3 most important matching tags
                    important_tags = [tag for tag, weight in tag_weights[:3]]
                    return f"Shares {', '.join(important_tags)}"
                else:
                    return "Similar tag profile"

        return "Tag similarity"

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
            'tags': str(game['tags']) if pd.notna(game['tags']) else '',
            'rating': float(game['rating']) if pd.notna(game['rating']) else 0.0,
            'ratings_count': int(game['ratings_count']) if pd.notna(game['ratings_count']) else 0,
            'added_to_library': int(game['added']) if pd.notna(game['added']) else 0,
            'platforms': str(game['platforms']) if pd.notna(game['platforms']) else ''
        }

    def get_tag_analysis(self) -> Dict[str, Any]:
        """Get analysis of tag distribution and TF-IDF weights"""

        # Get feature importance across all games
        if hasattr(self.tag_tfidf_matrix, 'toarray'):
            # Sparse matrix
            feature_importance = np.array(self.tag_tfidf_matrix.mean(axis=0)).flatten()
        else:
            # Dense matrix
            feature_importance = np.mean(self.tag_tfidf_matrix, axis=0)

        # Get most/least important tags
        sorted_indices = np.argsort(feature_importance)

        most_important = [
            {
                'tag': self.feature_names[idx],
                'avg_tfidf_score': round(feature_importance[idx], 4)
            }
            for idx in sorted_indices[-10:][::-1]  # Top 10, descending
        ]

        least_important = [
            {
                'tag': self.feature_names[idx],
                'avg_tfidf_score': round(feature_importance[idx], 4)
            }
            for idx in sorted_indices[:10]  # Bottom 10
        ]

        # Count games with tags
        games_with_tags = 0
        if hasattr(self.tag_tfidf_matrix, 'toarray'):
            games_with_tags = np.sum(np.any(self.tag_tfidf_matrix.toarray() > 0, axis=1))
        else:
            games_with_tags = np.sum(np.any(self.tag_tfidf_matrix > 0, axis=1))

        return {
            'total_unique_tags': len(self.feature_names),
            'games_with_tags': int(games_with_tags),
            'games_without_tags': len(self.games_df) - int(games_with_tags),
            'tfidf_matrix_shape': self.tag_tfidf_matrix.shape,
            'most_important_tags': most_important,
            'least_important_tags': least_important
        }

    def debug_game_tags(self, game_id: str) -> Dict[str, Any]:
        """Debug: Show TF-IDF breakdown for a specific game"""
        if str(game_id) not in self.game_id_to_index:
            return {'error': f'Game {game_id} not found'}

        idx = self.game_id_to_index[str(game_id)]
        game = self.games_df.iloc[idx]

        # Get TF-IDF vector for this game
        if hasattr(self.tag_tfidf_matrix, 'toarray'):
            game_vector = self.tag_tfidf_matrix[idx].toarray().flatten()
        else:
            game_vector = self.tag_tfidf_matrix[idx]

        # Find non-zero TF-IDF scores
        non_zero_indices = np.where(game_vector > 0)[0]

        tag_scores = [
            {
                'tag': self.feature_names[idx],
                'tfidf_score': round(game_vector[idx], 4)
            }
            for idx in non_zero_indices
        ]

        # Sort by TF-IDF score (most important first)
        tag_scores.sort(key=lambda x: x['tfidf_score'], reverse=True)

        return {
            'game_id': str(game['id']),
            'name': game['name'],
            'raw_tags': str(game['tags']) if pd.notna(game['tags']) else '',
            'processed_tags': len(tag_scores),
            'top_tfidf_tags': tag_scores[:10]  # Show top 10 most important tags
        }


# Testing function
if __name__ == "__main__":
    # Add src to path for imports
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from utils.data_loader import RAWGDatasetLoader

    print("🏷️ Testing TagBasedRecommender")
    print("=" * 50)

    # Load data
    loader = RAWGDatasetLoader()
    df = loader.load_and_process_data()

    # Get popular games for testing
    popular_df = loader.get_engagement_based_popular_games(debug=False)

    print(f"📊 Testing on {len(popular_df):,} popular games")

    # Initialize recommender
    recommender = TagBasedRecommender(popular_df)

    # Test recommendations
    test_games = ["19103", "12020"]  # Half-Life 2, Left 4 Dead 2
    print(f"\n🧪 Testing with games: {test_games}")

    # Debug the input games first
    for game_id in test_games:
        debug_info = recommender.debug_game_tags(game_id)
        print(f"\n🔍 {debug_info['name']} TF-IDF Analysis:")
        print(f"   Raw tags: {debug_info['raw_tags'][:100]}...")
        print(f"   Top TF-IDF tags:")
        for tag_info in debug_info['top_tfidf_tags'][:5]:
            print(f"     {tag_info['tag']}: {tag_info['tfidf_score']}")

    # Get recommendations
    recommendations = recommender.recommend(test_games, n=5)

    print(f"\n🎯 Top 5 Tag-Based Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Tags: {rec['tags']}")
        print(f"   Similarity: {rec['similarity_score']:.3f}")
        print(f"   Reason: {rec['reason']}")
        print()

    # Show tag analysis
    analysis = recommender.get_tag_analysis()
    print(f"\n📈 Tag Analysis:")
    print(f"Total unique tags processed: {analysis['total_unique_tags']}")
    print(f"Games with tags: {analysis['games_with_tags']:,}")
    print(f"TF-IDF matrix shape: {analysis['tfidf_matrix_shape']}")
    print(f"\nMost important tags (highest avg TF-IDF):")
    for tag_info in analysis['most_important_tags'][:5]:
        print(f"  {tag_info['tag']}: {tag_info['avg_tfidf_score']}")
