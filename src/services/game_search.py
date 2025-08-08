import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GameSearchService:
    def __init__(self, games_df: pd.DataFrame, game_id_to_index: Dict[str, int]):
        """Initialize with data from ContentBasedRecommender"""
        self.games_df = games_df
        self.game_id_to_index = game_id_to_index
        logger.info(f"🔍 GameSearchService initialized with {len(games_df):,} games")

    def search_games_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced fuzzy search with better matching"""
        from difflib import SequenceMatcher

        query_lower = query.lower().strip()
        query_words = query_lower.split()

        games_with_scores = []
        for idx, game in self.games_df.iterrows():
            name_lower = str(game['name']).lower()
            name_words = name_lower.split()

            # Multiple matching strategies
            score = 0

            # 1. Exact match
            if query_lower == name_lower:
                score = 100

            # 2. Starts with
            elif name_lower.startswith(query_lower):
                score = 90

            # 3. String similarity (handles typos)
            elif len(query_lower) >= 3:  # Only for longer queries
                similarity = SequenceMatcher(None, query_lower, name_lower).ratio()
                if similarity >= 0.8:  # 80% similarity
                    score = 80 + (similarity - 0.8) * 50  # Scale 80-90
                elif similarity >= 0.6:  # 60% similarity
                    score = 60 + (similarity - 0.6) * 50  # Scale 60-70

            # 4. Contains query
            elif query_lower in name_lower:
                score = 70 + (len(query_lower) / len(name_lower)) * 20

            # 5. Word matching with minimum threshold
            elif len(query_words) >= 2:  # Only for multi-word queries
                matched_words = sum(1 for word in query_words if word in name_words)
                match_ratio = matched_words / len(query_words)

                if match_ratio >= 0.6:  # At least 60% of words match
                    score = 40 + (match_ratio * 30)  # Scale 40-70

            # 6. Partial word matching (for typos like "minekraft")
            elif len(query_lower) >= 4:
                partial_matches = sum(1 for name_word in name_words
                                      if any(SequenceMatcher(None, query_lower, name_word).ratio() >= 0.7
                                             for name_word in name_words))
                if partial_matches > 0:
                    score = 30 + (partial_matches * 10)

            if score > 0:
                games_with_scores.append({
                    'game_id': str(game['id']),
                    'name': game['name'],
                    'description': game['description_raw'][:500] + '...' if len(str(game['description_raw'])) > 500 else str(game['description_raw']),
                    'genres': game['genres'],
                    'tags': game['tags'],
                    'platforms': game['platforms'],
                    'rating': float(game['rating']),
                    'ratings_count': int(game['ratings_count']),
                    'added_to_library': int(game['added']),
                    'release_date': str(game['released'])[:10] if pd.notna(game['released']) else None,
                    'metacritic': int(game['metacritic']) if game['metacritic'] > 0 else None,
                    'developers': game['developers'],
                    'publishers': game['publishers'],
                    'match_score': score  # Keep this for search ranking
                })

        # Sort by match score and return top results
        games_with_scores.sort(key=lambda x: x['match_score'], reverse=True)

        logger.info(f"🔍 Found {len(games_with_scores)} games matching '{query}'")
        return games_with_scores[:limit]

    def get_game_by_id(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed game info by ID"""
        if str(game_id) not in self.game_id_to_index:
            return None

        idx = self.game_id_to_index[str(game_id)]
        game = self.games_df.iloc[idx]

        return {
            'game_id': str(game['id']),
            'name': game['name'],
            'description': game['description_raw'][:500] + '...' if len(str(game['description_raw'])) > 500 else str(game['description_raw']),
            'genres': game['genres'],
            'tags': game['tags'],
            'platforms': game['platforms'],
            'rating': float(game['rating']),
            'ratings_count': int(game['ratings_count']),
            'added_to_library': int(game['added']),
            'release_date': str(game['released'])[:10] if pd.notna(game['released']) else None,
            'metacritic': int(game['metacritic']) if game['metacritic'] > 0 else None,
            'developers': game['developers'],
            'publishers': game['publishers']
        }
