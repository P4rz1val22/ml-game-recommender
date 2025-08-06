from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationRequest(BaseModel):
    """Request model for game recommendations"""
    liked_games: List[str] = Field(
        ...,
        description="List of app IDs for games the user has liked",
        example=["7940", "730", "440"]  # COD4, CS:GO, TF2
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of recommendations to return"
    )
    exclude_owned: bool = Field(
        default=True,
        description="Whether to exclude games the user already owns"
    )


class GameRecommendation(BaseModel):
    """Individual game recommendation"""
    app_id: str
    name: str
    genres: str
    tags: str
    positive_reviews: int
    negative_reviews: int
    estimated_owners: str
    price: float
    similarity_score: float
    reason: str


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[GameRecommendation]
    total_found: int
    algorithm: str = "content_based_popular"
    processing_time_ms: Optional[float] = None


class GameInfo(BaseModel):
    """Detailed game information"""
    app_id: str
    name: str
    description: str
    genres: str
    price: float
    positive_reviews: int
    estimated_owners: str
