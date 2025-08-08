from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict


class RecommendationRequest(BaseModel):
    """Request model for standard game recommendations"""
    liked_games: List[str] = Field(
        ...,
        description="List of game IDs for games the user has liked",
        example=["19103", "12020"]  # Half-Life 2, Left 4 Dead 2
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


class ModelWeights(BaseModel):
    """Configurable weights for hybrid recommendations"""
    description: float = Field(ge=0, le=1, default=0.6, description="Weight for description-based similarity")
    genre: float = Field(ge=0, le=1, default=0.3, description="Weight for genre-based similarity")
    tags: float = Field(ge=0, le=1, default=0.1, description="Weight for tag-based similarity")

    @field_validator('tags')
    @classmethod
    def validate_weights_sum(cls, v, info):
        # This runs after all fields are set
        if info.data:
            total = info.data.get('description', 0) + info.data.get('genre', 0) + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f'Weights must sum to 1.0, got {total:.3f}')
        return v


class HybridRecommendationRequest(BaseModel):
    """Request model for hybrid recommendations with configurable weights"""
    liked_games: List[str] = Field(..., description="List of game IDs that the user likes")
    limit: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    exclude_owned: bool = Field(default=True, description="Whether to exclude games the user already owns")
    weights: Optional[ModelWeights] = Field(default=None, description="Custom model weights (defaults to balanced)")


class GameRecommendation(BaseModel):
    """Individual game recommendation with optional model breakdown"""
    game_id: str
    name: str
    description: str
    release_date: str
    genres: str
    tags: str
    rating: float
    ratings_count: int
    added_to_library: int
    platforms: str
    similarity_score: float
    reason: str
    model_breakdown: Optional[Dict[str, float]] = None  # Individual model scores for hybrid


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[GameRecommendation]
    total_found: int
    algorithm: str
    processing_time_ms: float


class GameDetail(BaseModel):
    """Detailed game information - unified format"""
    game_id: str
    name: str
    description: str
    genres: str
    tags: str
    platforms: str
    rating: float
    ratings_count: int
    added_to_library: int
    release_date: Optional[str]
    metacritic: Optional[int]
    developers: str
    publishers: str


class GameInfo(BaseModel):
    """Basic game information"""
    game_id: str
    name: str
    description: str
    genres: str
    rating: float
    ratings_count: int
    added_to_library: int
    platforms: str
