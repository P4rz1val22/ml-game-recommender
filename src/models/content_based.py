from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'cache'):
        self.model_name = model_name
        self.sentence_model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)  # Create cache directory

        # Cache file paths
        self.embeddings_cache = self.cache_dir / f"popular_embeddings_{model_name.replace('/', '_')}.npy"
        self.games_cache = self.cache_dir / f"popular_games_{model_name.replace('/', '_')}.pkl"
        self.mapping_cache = self.cache_dir / f"app_id_mapping_{model_name.replace('/', '_')}.pkl"

        logger.info(f"🧠 Initializing ContentBasedRecommender with {model_name}")

    def _load_sentence_model(self):
        """Lazy load the sentence transformer model"""
        if self.sentence_model is None:
            logger.info(f"📥 Loading sentence transformer: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
            logger.info("✅ Sentence transformer loaded successfully")

    def _load_from_cache(self) -> bool:
        """Try to load embeddings from cache"""
        try:
            if (self.embeddings_cache.exists() and
                    self.games_cache.exists() and
                    self.mapping_cache.exists()):
                logger.info("📂 Loading cached embeddings...")
                self.popular_embeddings = np.load(self.embeddings_cache)
                self.popular_games = pd.read_pickle(self.games_cache)

                import pickle
                with open(self.mapping_cache, 'rb') as f:
                    self.app_id_to_index = pickle.load(f)

                return True
        except Exception as e:
            logger.warning(f"⚠️ Cache loading failed: {e}")
            return False

        return False

    def _save_to_cache(self):
        """Save embeddings to cache"""
        try:
            logger.info("💾 Saving embeddings to cache...")
            np.save(self.embeddings_cache, self.popular_embeddings)
            self.popular_games.to_pickle(self.games_cache)

            import pickle
            with open(self.mapping_cache, 'wb') as f:
                pickle.dump(self.app_id_to_index, f)

            logger.info("✅ Cache saved successfully!")
        except Exception as e:
            logger.error(f"❌ Cache saving failed: {e}")

    def fit_popular_games(self, games_df: pd.DataFrame):
        """Fit the model on popular games with caching"""
        logger.info("🔥 Fitting model on popular games...")

        # Filter to popular games only
        self.popular_games = games_df[
            ~games_df['estimated_owners'].isin(['0 - 20000', '0 - 0'])
        ].copy()
        logger.info(f"📊 Popular games dataset: {len(self.popular_games):,} games")

        # Check if we have cached embeddings
        if self._load_from_cache():
            logger.info("🚀 Loaded embeddings from cache - SUPER FAST!")
            return

        # If no cache, generate embeddings (slow path)
        logger.info("💾 No cache found, generating embeddings (this will take a few minutes)...")
        self._load_sentence_model()

        descriptions = self.popular_games['detailed_description'].fillna('').tolist()
        logger.info("🔤 Generating embeddings for game descriptions...")

        self.popular_embeddings = self.sentence_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32
        )

        # Create app_id mapping
        self.app_id_to_index = {
            str(app_id): idx for idx, app_id in enumerate(self.popular_games['app_id'])
        }

        # Save to cache for next time
        self._save_to_cache()

        logger.info(f"✅ Model fitted and cached successfully!")
        logger.info(f"📐 Embeddings shape: {self.popular_embeddings.shape}")

    def _get_user_preference_vector(self, liked_app_ids: List[str]) -> Optional[np.ndarray]:
        """Calculate user preference vector from liked games"""
        valid_embeddings = []
        found_games = []

        for app_id in liked_app_ids:
            if str(app_id) in self.app_id_to_index:
                idx = self.app_id_to_index[str(app_id)]
                valid_embeddings.append(self.popular_embeddings[idx])
                found_games.append(self.popular_games.iloc[idx]['name'])

        if not valid_embeddings:
            logger.warning(f"❌ No valid games found in popular dataset from: {liked_app_ids}")
            return None

        logger.info(f"✅ Found {len(valid_embeddings)} games: {found_games}")

        # Average the embeddings to create user preference vector
        user_preference = np.mean(valid_embeddings, axis=0)
        return user_preference

    def recommend_popular(
            self,
            liked_app_ids: List[str],
            n: int = 10,
            exclude_owned: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate recommendations from popular games"""

        if self.popular_embeddings is None:
            raise ValueError("Model not fitted! Call fit_popular_games() first.")

        # Get user preference vector
        user_preference = self._get_user_preference_vector(liked_app_ids)
        if user_preference is None:
            return []

        # Calculate similarities to all popular games
        similarities = cosine_similarity([user_preference], self.popular_embeddings)[0]

        # Get top similar games
        if exclude_owned:
            # Exclude games user already owns
            owned_indices = [self.app_id_to_index[str(app_id)]
                             for app_id in liked_app_ids
                             if str(app_id) in self.app_id_to_index]
            similarities[owned_indices] = -1  # Set to -1 so they won't be selected

        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:n]

        # Format recommendations
        recommendations = []
        for idx in top_indices:
            if similarities[idx] <= 0:  # Skip excluded or irrelevant games
                continue

            game = self.popular_games.iloc[idx]
            recommendations.append({
                'app_id': str(game['app_id']),
                'name': game['name'],
                'genres': game['genres'],
                'tags': game['tags'][:100] + '...' if len(game['tags']) > 100 else game['tags'],
                'positive_reviews': int(game['positive']),
                'negative_reviews': int(game['negative']),
                'estimated_owners': game['estimated_owners'],
                'price': float(game['price']),
                'similarity_score': float(similarities[idx]),
                'reason': f"Similar to games you liked"
            })

        logger.info(f"🎯 Generated {len(recommendations)} recommendations")
        return recommendations[:n]

    def get_game_info(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific game"""
        if str(app_id) not in self.app_id_to_index:
            return None

        idx = self.app_id_to_index[str(app_id)]
        game = self.popular_games.iloc[idx]

        return {
            'app_id': str(game['app_id']),
            'name': game['name'],
            'description': game['detailed_description'][:200] + '...',
            'genres': game['genres'],
            'price': float(game['price']),
            'positive_reviews': int(game['positive']),
            'estimated_owners': game['estimated_owners']
        }
