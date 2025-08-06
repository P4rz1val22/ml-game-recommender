from fastapi import APIRouter, HTTPException
import time
import logging

from api.models import RecommendationRequest, RecommendationResponse, GameRecommendation
from models.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)
router = APIRouter()

# Global recommender instance (loaded once at startup)
recommender: ContentBasedRecommender = None


def initialize_recommender(games_df):
    """Initialize the global recommender instance"""
    global recommender
    logger.info("🤖 Initializing recommendation engine...")
    recommender = ContentBasedRecommender()
    recommender.fit_popular_games(games_df)
    logger.info("✅ Recommendation engine ready!")


@router.post("/popular", response_model=RecommendationResponse)
async def get_popular_recommendations(request: RecommendationRequest):
    """Get recommendations from popular games"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    start_time = time.time()

    try:
        recommendations = recommender.recommend_popular(
            liked_app_ids=request.liked_games,
            n=request.limit,
            exclude_owned=request.exclude_owned
        )

        # Convert to response format
        game_recs = [
            GameRecommendation(**rec) for rec in recommendations
        ]

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return RecommendationResponse(
            recommendations=game_recs,
            total_found=len(game_recs),
            algorithm="content_based_popular",
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
