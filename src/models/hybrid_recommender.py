import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HybridRecommender:
    def __init__(self, content_recommender, genre_recommender, tag_recommender):
        """Initialize hybrid recommender with all three models"""
        self.content_recommender = content_recommender
        self.genre_recommender = genre_recommender
        self.tag_recommender = tag_recommender

        # Use content recommender's data as the base (it has the most complete info)
        self.games_df = content_recommender.popular_games
        self.game_id_to_index = content_recommender.game_id_to_index

        logger.info(f"🔥 HybridRecommender initialized with {len(self.games_df):,} games")

    def get_similarities(
            self,
            liked_game_ids: List[str],
            weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Get hybrid similarity scores combining all three models"""

        # Default weights
        if weights is None:
            weights = {"description": 0.6, "genre": 0.3, "tags": 0.1}

        # Validate weights sum to 1.0 (allow small floating point errors)
        total_weight = sum(weights.values())
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        logger.info(f"🎚️ Using weights: {weights}")

        # Get similarity scores from each model
        desc_similarities = self.content_recommender.get_similarities(liked_game_ids)
        genre_similarities = self.genre_recommender.get_similarities(liked_game_ids)
        tag_similarities = self.tag_recommender.get_similarities(liked_game_ids)

        # Weighted combination
        final_similarities = (
                weights["description"] * desc_similarities +
                weights["genre"] * genre_similarities +
                weights["tags"] * tag_similarities
        )

        return final_similarities

    def recommend_hybrid(
            self,
            liked_game_ids: List[str],
            weights: Optional[Dict[str, float]] = None,
            n: int = 10,
            exclude_owned: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations combining all three models"""

        # Get combined similarity scores
        similarities = self.get_similarities(liked_game_ids, weights)

        if np.all(similarities == 0):
            logger.warning("No similarities found across any model")
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
        top_indices = np.argsort(similarities)[::-1][:n * 2]  # Buffer for filtering

        recommendations = []
        for idx in top_indices:
            if similarities[idx] <= 0:  # Skip excluded or zero similarity
                continue

            game = self.games_df.iloc[idx]

            # Get explanations from each model
            hybrid_explanation = self._get_hybrid_explanation(
                liked_game_ids, str(game['id']), weights or {"description": 0.6, "genre": 0.3, "tags": 0.1}
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
                'reason': f"Hybrid match: {hybrid_explanation}",
                'model_breakdown': self._get_model_breakdown(liked_game_ids, str(game['id']))
            })

            if len(recommendations) >= n:
                break

        logger.info(f"🎯 Generated {len(recommendations)} hybrid recommendations")
        return recommendations

    def _get_hybrid_explanation(
            self,
            liked_game_ids: List[str],
            recommended_game_id: str,
            weights: Dict[str, float]
    ) -> str:
        """Create explanation showing which model contributed most"""

        # Get individual similarities for this specific game
        if str(recommended_game_id) not in self.game_id_to_index:
            return "Multi-model similarity"

        idx = self.game_id_to_index[str(recommended_game_id)]

        desc_score = self.content_recommender.get_similarities(liked_game_ids)[idx]
        genre_score = self.genre_recommender.get_similarities(liked_game_ids)[idx]
        tag_score = self.tag_recommender.get_similarities(liked_game_ids)[idx]

        # Calculate weighted contributions
        desc_contribution = weights["description"] * desc_score
        genre_contribution = weights["genre"] * genre_score
        tag_contribution = weights["tags"] * tag_score

        # Find the strongest contributor
        contributions = [
            ("description", desc_contribution),
            ("genre", genre_contribution),
            ("tags", tag_contribution)
        ]

        # Sort by contribution strength
        contributions.sort(key=lambda x: x[1], reverse=True)
        primary_model = contributions[0][0]

        # Create explanation based on strongest model
        if primary_model == "description":
            return "Semantic themes + genre/tag support"
        elif primary_model == "genre":
            return "Genre match + content/tag support"
        else:
            return "Tag mechanics + genre/content support"

    def _get_model_breakdown(self, liked_game_ids: List[str], recommended_game_id: str) -> Dict[str, float]:
        """Get individual model scores for transparency"""
        if str(recommended_game_id) not in self.game_id_to_index:
            return {"description": 0.0, "genre": 0.0, "tags": 0.0}

        idx = self.game_id_to_index[str(recommended_game_id)]

        return {
            "description": float(self.content_recommender.get_similarities(liked_game_ids)[idx]),
            "genre": float(self.genre_recommender.get_similarities(liked_game_ids)[idx]),
            "tags": float(self.tag_recommender.get_similarities(liked_game_ids)[idx])
        }

    def get_game_info(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific game"""
        return self.content_recommender.get_game_info(game_id)


# Testing function
if __name__ == "__main__":
    # Add src to path for imports
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))

    from utils.data_loader import RAWGDatasetLoader
    from models.content_based import ContentBasedRecommender
    from models.genre_based import GenreBasedRecommender
    from models.tag_based import TagBasedRecommender

    print("🔥 Testing HybridRecommender")
    print("=" * 50)

    # Load data
    loader = RAWGDatasetLoader()
    df = loader.load_and_process_data()
    popular_df = loader.get_engagement_based_popular_games(debug=False)

    print(f"📊 Testing on {len(popular_df):,} popular games")

    # Initialize all models
    print("🧠 Loading models...")
    content_rec = ContentBasedRecommender()
    content_rec.fit_popular_games(popular_df)

    genre_rec = GenreBasedRecommender(popular_df)
    tag_rec = TagBasedRecommender(popular_df)

    # Initialize hybrid
    hybrid_rec = HybridRecommender(content_rec, genre_rec, tag_rec)

    # Test different weight combinations
    test_games = ["19103", "12020"]  # Half-Life 2, Left 4 Dead 2

    weight_scenarios = [
        {"description": 0.6, "genre": 0.3, "tags": 0.1},  # Balanced (default)
        {"description": 0.8, "genre": 0.1, "tags": 0.1},  # Description-heavy
        {"description": 0.2, "genre": 0.6, "tags": 0.2},  # Genre-heavy
        {"description": 0.2, "genre": 0.2, "tags": 0.6},  # Tag-heavy
    ]

    for i, weights in enumerate(weight_scenarios, 1):
        print(f"\n🧪 Test {i}: {weights}")
        recommendations = hybrid_rec.recommend_hybrid(test_games, weights=weights, n=3)

        print(f"Top 3 Hybrid Recommendations:")
        for j, rec in enumerate(recommendations, 1):
            print(f"  {j}. {rec['name']}")
            print(f"     Similarity: {rec['similarity_score']:.3f}")
            print(f"     Reason: {rec['reason']}")
            print(f"     Breakdown: {rec['model_breakdown']}")
