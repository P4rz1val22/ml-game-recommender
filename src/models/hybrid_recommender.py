import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)


class FranchiseDetector:
    """Intelligent franchise detection system for games"""

    def __init__(self):
        # Common franchise patterns and separators
        self.sequel_patterns = [
            r'\s*\d+$',  # "Game 2", "Game 3"
            r'\s*II+$',  # "Game II", "Game III"
            r'\s*IV+$',  # "Game IV", "Game V"
            r':\s*.+$',  # "Game: Subtitle"
            r'\s*-\s*.+$',  # "Game - Subtitle"
            r'\s*Episode\s*.+$',  # "Game Episode 1"
            r'\s*Chapter\s*.+$',  # "Game Chapter 1"
            r'\s*Part\s*.+$',  # "Game Part 2"
            r'\s*\(.+\)$',  # "Game (Year)" or "Game (Edition)"
        ]

    def extract_franchise_base(self, game_name: str) -> str:
        """Extract the base franchise name from a game title"""
        if not game_name or pd.isna(game_name):
            return ""

        name = str(game_name).strip()

        # Remove common sequel patterns
        for pattern in self.sequel_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        # Clean up remaining whitespace
        name = name.strip()

        # Handle special cases where the above patterns might be too aggressive
        # If we end up with something too short, use original name
        if len(name) < 3:
            return str(game_name).strip()

        return name

    def find_franchise_games(self, liked_games: List[str], games_df: pd.DataFrame, game_id_to_index: Dict[str, int]) -> Set[int]:
        """Find all games that belong to the same franchises as liked games"""
        franchise_indices = set()

        # Extract franchise bases from liked games
        franchise_bases = set()
        for game_id in liked_games:
            if str(game_id) in game_id_to_index:
                idx = game_id_to_index[str(game_id)]
                game_name = games_df.iloc[idx]['name']
                franchise_base = self.extract_franchise_base(game_name)
                if franchise_base and len(franchise_base) >= 3:  # Avoid too-short bases
                    franchise_bases.add(franchise_base.lower())

        if not franchise_bases:
            return franchise_indices

        # Find all games matching these franchise bases
        for idx, game_name in enumerate(games_df['name']):
            if pd.notna(game_name):
                game_franchise = self.extract_franchise_base(str(game_name)).lower()
                if game_franchise in franchise_bases:
                    franchise_indices.add(idx)

        return franchise_indices


class GameFilterSystem:
    """Comprehensive game filtering system"""

    def __init__(self):
        self.franchise_detector = FranchiseDetector()

    def apply_filters(
            self,
            similarities: np.ndarray,
            games_df: pd.DataFrame,
            game_id_to_index: Dict[str, int],
            liked_games: List[str],
            filters: Dict[str, Any]
    ) -> np.ndarray:
        """Apply all filters to similarity scores"""

        filtered_similarities = similarities.copy()
        filter_stats = {"total_games": len(similarities), "filtered_counts": {}}

        # 1. Franchise filtering
        if filters.get('exclude_franchise', False):
            franchise_indices = self.franchise_detector.find_franchise_games(
                liked_games, games_df, game_id_to_index
            )
            for idx in franchise_indices:
                filtered_similarities[idx] = -1
            filter_stats["filtered_counts"]["franchise"] = len(franchise_indices)

        # 2. Platform filtering
        if filters.get('platforms'):
            platform_indices = self._filter_by_platforms(
                games_df, filters['platforms']
            )
            for idx in platform_indices:
                filtered_similarities[idx] = -1
            filter_stats["filtered_counts"]["platform"] = len(platform_indices)

        # 3. Date range filtering
        if filters.get('min_year') or filters.get('max_year'):
            date_indices = self._filter_by_date_range(
                games_df, filters.get('min_year'), filters.get('max_year')
            )
            for idx in date_indices:
                filtered_similarities[idx] = -1
            filter_stats["filtered_counts"]["date_range"] = len(date_indices)

        # 4. Quality filtering
        if filters.get('min_rating') or filters.get('min_reviews'):
            quality_indices = self._filter_by_quality(
                games_df, filters.get('min_rating'), filters.get('min_reviews')
            )
            for idx in quality_indices:
                filtered_similarities[idx] = -1
            filter_stats["filtered_counts"]["quality"] = len(quality_indices)

        # Log filtering statistics
        total_filtered = np.sum(filtered_similarities == -1)
        remaining = np.sum(filtered_similarities > 0)

        logger.info(f"🔍 Applied filters: {filter_stats['filtered_counts']}")
        logger.info(f"📊 Filtered {total_filtered:,} games, {remaining:,} candidates remaining")

        return filtered_similarities

    def _filter_by_platforms(self, games_df: pd.DataFrame, allowed_platforms: List[str]) -> Set[int]:
        """Filter games that don't match specified platforms"""
        excluded_indices = set()

        # Normalize platform names for matching
        allowed_platforms_lower = [p.lower() for p in allowed_platforms]

        for idx, platforms_str in enumerate(games_df['platforms']):
            if pd.notna(platforms_str):
                game_platforms = str(platforms_str).lower()

                # Check if any allowed platform is in this game's platforms
                has_allowed_platform = any(
                    platform in game_platforms for platform in allowed_platforms_lower
                )

                if not has_allowed_platform:
                    excluded_indices.add(idx)
            else:
                # Exclude games with no platform information
                excluded_indices.add(idx)

        return excluded_indices

    def _filter_by_date_range(self, games_df: pd.DataFrame, min_year: Optional[int], max_year: Optional[int]) -> Set[int]:
        """Filter games outside the specified date range"""
        excluded_indices = set()

        for idx, release_date in enumerate(games_df['released']):
            if pd.notna(release_date):
                try:
                    # Extract year from release date
                    if isinstance(release_date, str):
                        release_year = int(release_date[:4])
                    else:
                        release_year = release_date.year

                    # Check date range constraints
                    if min_year and release_year < min_year:
                        excluded_indices.add(idx)
                    elif max_year and release_year > max_year:
                        excluded_indices.add(idx)

                except (ValueError, AttributeError):
                    # Exclude games with invalid date formats
                    excluded_indices.add(idx)
            else:
                # Exclude games with no release date
                excluded_indices.add(idx)

        return excluded_indices

    def _filter_by_quality(self, games_df: pd.DataFrame, min_rating: Optional[float], min_reviews: Optional[int]) -> Set[int]:
        """Filter games below quality thresholds"""
        excluded_indices = set()

        for idx, (rating, review_count) in enumerate(zip(games_df['rating'], games_df['ratings_count'])):
            # Rating filter
            if min_rating and (pd.isna(rating) or float(rating) < min_rating):
                excluded_indices.add(idx)

            # Review count filter
            if min_reviews and (pd.isna(review_count) or int(review_count) < min_reviews):
                excluded_indices.add(idx)

        return excluded_indices

    def get_filter_explanation(self, filters: Dict[str, Any]) -> str:
        """Generate explanation text for applied filters"""
        explanations = []

        if filters.get('exclude_franchise'):
            explanations.append("franchise filtered")

        if filters.get('platforms'):
            platform_list = ", ".join(filters['platforms'][:2])  # Show first 2 platforms
            if len(filters['platforms']) > 2:
                platform_list += f" +{len(filters['platforms']) - 2} more"
            explanations.append(f"platforms: {platform_list}")

        if filters.get('min_year') or filters.get('max_year'):
            if filters.get('min_year') and filters.get('max_year'):
                explanations.append(f"years: {filters['min_year']}-{filters['max_year']}")
            elif filters.get('min_year'):
                explanations.append(f"from {filters['min_year']}+")
            else:
                explanations.append(f"before {filters['max_year']}")

        if filters.get('min_rating'):
            explanations.append(f"rating ≥{filters['min_rating']}")

        if filters.get('min_reviews'):
            explanations.append(f"≥{filters['min_reviews']} reviews")

        if explanations:
            return f" ({', '.join(explanations)})"
        return ""


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

        logger.info(f"Using weights: {weights}")

        if hasattr(self, '_cached_desc_similarities'):
            desc_similarities = self._cached_desc_similarities
            genre_similarities = self._cached_genre_similarities
            tag_similarities = self._cached_tag_similarities
        else:
            desc_similarities = self.content_recommender.get_similarities(liked_game_ids)
            genre_similarities = self.genre_recommender.get_similarities(liked_game_ids)
            tag_similarities = self.tag_recommender.get_similarities(liked_game_ids)

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
            exclude_owned: bool = True,
            exclude_franchise: bool = False,  # Legacy support
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations with comprehensive filtering"""

        # Clear any previous cached similarities
        self._clear_similarity_cache()

        # Get individual model similarities (cache these for model breakdown)
        logger.info("🔄 Computing individual model similarities...")
        self._cached_desc_similarities = self.content_recommender.get_similarities(liked_game_ids)
        self._cached_genre_similarities = self.genre_recommender.get_similarities(liked_game_ids)
        self._cached_tag_similarities = self.tag_recommender.get_similarities(liked_game_ids)

        # Get combined similarity scores using cached results
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

        # Prepare filters dictionary
        if filters is None:
            filters = {}

        # Legacy support: merge exclude_franchise into filters
        if exclude_franchise:
            filters['exclude_franchise'] = True

        # Apply comprehensive filtering
        filter_explanation = ""
        if filters:
            filter_system = GameFilterSystem()
            similarities = filter_system.apply_filters(
                similarities, self.games_df, self.game_id_to_index, liked_game_ids, filters
            )
            filter_explanation = filter_system.get_filter_explanation(filters)

        # Get top N recommendations (larger buffer due to more filtering)
        top_indices = np.argsort(similarities)[::-1][:n * 3]

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
                'reason': f"Hybrid match: {hybrid_explanation}{filter_explanation}",
                'model_breakdown': self._get_model_breakdown_cached(str(game['id']))
            })

            if len(recommendations) >= n:
                break

        # Clear cache after use to free memory
        self._clear_similarity_cache()

        logger.info(f"🎯 Generated {len(recommendations)} hybrid recommendations with filtering")
        return recommendations

    def _get_hybrid_explanation(
            self,
            liked_game_ids: List[str],
            recommended_game_id: str,
            weights: Dict[str, float]
    ) -> str:
        """Create explanation showing which model contributed most (OPTIMIZED)"""

        # Get individual similarities for this specific game
        if str(recommended_game_id) not in self.game_id_to_index:
            return "Multi-model similarity"

        idx = self.game_id_to_index[str(recommended_game_id)]

        # Use cached similarities instead of recalculating
        desc_score = self._cached_desc_similarities[idx]
        genre_score = self._cached_genre_similarities[idx]
        tag_score = self._cached_tag_similarities[idx]

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

    def _clear_similarity_cache(self):
        """Clear cached similarity scores to free memory"""
        if hasattr(self, '_cached_desc_similarities'):
            delattr(self, '_cached_desc_similarities')
        if hasattr(self, '_cached_genre_similarities'):
            delattr(self, '_cached_genre_similarities')
        if hasattr(self, '_cached_tag_similarities'):
            delattr(self, '_cached_tag_similarities')

    def _get_model_breakdown_cached(self, recommended_game_id: str) -> Dict[str, float]:
        """Get individual model scores using cached similarities (OPTIMIZED)"""
        if str(recommended_game_id) not in self.game_id_to_index:
            return {"description": 0.0, "genre": 0.0, "tags": 0.0}

        idx = self.game_id_to_index[str(recommended_game_id)]

        # Use cached similarities - no additional API calls!
        return {
            "description": float(self._cached_desc_similarities[idx]),
            "genre": float(self._cached_genre_similarities[idx]),
            "tags": float(self._cached_tag_similarities[idx])
        }

    def _get_model_breakdown(self, liked_game_ids: List[str], recommended_game_id: str) -> Dict[str, float]:
        """Get individual model scores for transparency (LEGACY - less efficient)"""
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
