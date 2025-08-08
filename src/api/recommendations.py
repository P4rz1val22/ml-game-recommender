from fastapi import APIRouter, HTTPException
import time
import logging

from api.models import RecommendationRequest, RecommendationResponse, GameRecommendation, HybridRecommendationRequest
from models.content_based import ContentBasedRecommender
from models.genre_based import GenreBasedRecommender
from models.tag_based import TagBasedRecommender
from models.hybrid_recommender import HybridRecommender

logger = logging.getLogger(__name__)
router = APIRouter()

# Global recommender instances (loaded once at startup)
content_recommender: ContentBasedRecommender = None
genre_recommender: GenreBasedRecommender = None
tag_recommender: TagBasedRecommender = None
hybrid_recommender: HybridRecommender = None


def initialize_recommender(processed_df, loader):
    """Initialize all global recommender instances with RAWG data"""
    global content_recommender, genre_recommender, tag_recommender, hybrid_recommender
    logger.info("🤖 Initializing RAWG recommendation engines...")

    # Get engagement-based popular games
    popular_df = loader.get_engagement_based_popular_games(debug=False)
    logger.info(f"📊 Using {len(popular_df):,} popular games for all models")

    # Initialize all three individual models
    logger.info("🧠 Loading ContentBased model...")
    content_recommender = ContentBasedRecommender()
    content_recommender.fit_popular_games(popular_df)

    logger.info("🎨 Loading GenreBased model...")
    genre_recommender = GenreBasedRecommender(popular_df)

    logger.info("🏷️ Loading TagBased model...")
    tag_recommender = TagBasedRecommender(popular_df)

    # Initialize hybrid model
    logger.info("🔥 Loading HybridRecommender...")
    hybrid_recommender = HybridRecommender(content_recommender, genre_recommender, tag_recommender)

    logger.info("✅ All RAWG recommendation engines ready!")


@router.post("/description", response_model=RecommendationResponse)
async def get_description_recommendations(request: RecommendationRequest):
    """Get recommendations based on game descriptions using semantic similarity"""
    if content_recommender is None:
        raise HTTPException(status_code=503, detail="Content recommendation engine not initialized")

    start_time = time.time()

    try:
        recommendations = content_recommender.recommend_popular(
            liked_game_ids=request.liked_games,
            n=request.limit,
            exclude_owned=request.exclude_owned
        )

        # Convert to response format
        game_recs = [
            GameRecommendation(**rec) for rec in recommendations
        ]

        processing_time = (time.time() - start_time) * 1000

        return RecommendationResponse(
            recommendations=game_recs,
            total_found=len(game_recs),
            algorithm="content_based_description",
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Description recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Description recommendation failed: {str(e)}")


@router.post("/genre", response_model=RecommendationResponse)
async def get_genre_recommendations(request: RecommendationRequest):
    """Get recommendations based on genre similarity with rarity weighting"""
    if genre_recommender is None:
        raise HTTPException(status_code=503, detail="Genre recommendation engine not initialized")

    start_time = time.time()

    try:
        recommendations = genre_recommender.recommend(
            liked_game_ids=request.liked_games,
            n=request.limit,
            exclude_owned=request.exclude_owned
        )

        # Convert to response format
        game_recs = [
            GameRecommendation(**rec) for rec in recommendations
        ]

        processing_time = (time.time() - start_time) * 1000

        return RecommendationResponse(
            recommendations=game_recs,
            total_found=len(game_recs),
            algorithm="genre_based_rarity_weighted",
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Genre recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Genre recommendation failed: {str(e)}")


@router.post("/tags", response_model=RecommendationResponse)
async def get_tag_recommendations(request: RecommendationRequest):
    """Get recommendations based on tag similarity using TF-IDF"""
    if tag_recommender is None:
        raise HTTPException(status_code=503, detail="Tag recommendation engine not initialized")

    start_time = time.time()

    try:
        recommendations = tag_recommender.recommend(
            liked_game_ids=request.liked_games,
            n=request.limit,
            exclude_owned=request.exclude_owned
        )

        # Convert to response format
        game_recs = [
            GameRecommendation(**rec) for rec in recommendations
        ]

        processing_time = (time.time() - start_time) * 1000

        return RecommendationResponse(
            recommendations=game_recs,
            total_found=len(game_recs),
            algorithm="tag_based_tfidf",
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Tag recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Tag recommendation failed: {str(e)}")


@router.post("/hybrid", response_model=RecommendationResponse)
async def get_hybrid_recommendations(request: HybridRecommendationRequest):
    """Get recommendations using hybrid model with configurable weights"""
    if hybrid_recommender is None:
        raise HTTPException(status_code=503, detail="Hybrid recommendation engine not initialized")

    start_time = time.time()

    try:
        # Convert ModelWeights to dict if provided
        weights_dict = None
        if request.weights:
            weights_dict = {
                "description": request.weights.description,
                "genre": request.weights.genre,
                "tags": request.weights.tags
            }

        recommendations = hybrid_recommender.recommend_hybrid(
            liked_game_ids=request.liked_games,
            weights=weights_dict,
            n=request.limit,
            exclude_owned=request.exclude_owned
        )

        # Convert to response format
        game_recs = [
            GameRecommendation(**rec) for rec in recommendations
        ]

        processing_time = (time.time() - start_time) * 1000

        # Include weight info in algorithm name
        weight_info = weights_dict or {"description": 0.6, "genre": 0.3, "tags": 0.1}
        algorithm_name = f"hybrid_d{weight_info['description']}_g{weight_info['genre']}_t{weight_info['tags']}"

        return RecommendationResponse(
            recommendations=game_recs,
            total_found=len(game_recs),
            algorithm=algorithm_name,
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Hybrid recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid recommendation failed: {str(e)}")


# Debug/analysis endpoints
@router.get("/models/status")
async def get_models_status():
    """Get status of all recommendation models"""
    return {
        "content_model": "loaded" if content_recommender is not None else "not_loaded",
        "genre_model": "loaded" if genre_recommender is not None else "not_loaded",
        "tag_model": "loaded" if tag_recommender is not None else "not_loaded",
        "hybrid_model": "loaded" if hybrid_recommender is not None else "not_loaded",
        "total_models": sum([
            1 if content_recommender is not None else 0,
            1 if genre_recommender is not None else 0,
            1 if tag_recommender is not None else 0,
            1 if hybrid_recommender is not None else 0
        ])
    }


@router.get("/models/analysis")
async def get_models_analysis():
    """Get detailed analysis of all models"""
    analysis = {}

    if content_recommender is not None:
        analysis["content_model"] = {
            "algorithm": "sentence_transformers",
            "embeddings_shape": content_recommender.popular_embeddings.shape if content_recommender.popular_embeddings is not None else None,
            "games_count": len(content_recommender.popular_games) if content_recommender.popular_games is not None else 0
        }

    if genre_recommender is not None:
        analysis["genre_model"] = genre_recommender.get_genre_analysis()

    if tag_recommender is not None:
        analysis["tag_model"] = tag_recommender.get_tag_analysis()

    if hybrid_recommender is not None:
        analysis["hybrid_model"] = {
            "components": ["content_based", "genre_based", "tag_based"],
            "default_weights": {"description": 0.6, "genre": 0.3, "tags": 0.1}
        }

    return analysis
