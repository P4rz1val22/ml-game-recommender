from fastapi import APIRouter, HTTPException
import time
import logging

from api.models import RecommendationRequest, RecommendationResponse, GameRecommendation
from models.content_based import ContentBasedRecommender

logger = logging.getLogger(__name__)
router = APIRouter()

# Global recommender instance (loaded once at startup)
recommender: ContentBasedRecommender = None


def initialize_recommender(processed_df, loader):
    """Initialize the global recommender instance with RAWG data"""
    global recommender
    logger.info("🤖 Initializing RAWG recommendation engine...")

    # Get engagement-based popular games
    popular_df = loader.get_engagement_based_popular_games(debug=False)

    recommender = ContentBasedRecommender()
    recommender.fit_popular_games(popular_df)
    logger.info("✅ RAWG recommendation engine ready!")


@router.post("/popular", response_model=RecommendationResponse)
async def get_popular_recommendations(request: RecommendationRequest):
    """Get recommendations from popular games using RAWG data"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")

    start_time = time.time()

    try:
        recommendations = recommender.recommend_popular(
            liked_game_ids=request.liked_games,  # Updated parameter name
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
            algorithm="content_based_rawg_engagement",  # Updated algorithm name
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
