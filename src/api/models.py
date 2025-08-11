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


class GameFilters(BaseModel):
    """Advanced filtering options for game recommendations"""
    exclude_franchise: bool = Field(default=False, description="Exclude games from same franchise as liked games")
    platforms: Optional[List[str]] = Field(default=None, description="Filter by platforms (e.g., ['PC', 'PlayStation', 'Xbox'])")
    min_year: Optional[int] = Field(default=None, ge=1970, le=2030, description="Minimum release year (e.g., 2020)")
    max_year: Optional[int] = Field(default=None, ge=1970, le=2030, description="Maximum release year (e.g., 2015)")
    min_rating: Optional[float] = Field(default=None, ge=0.0, le=5.0, description="Minimum game rating (e.g., 3.5)")
    min_reviews: Optional[int] = Field(default=None, ge=1, description="Minimum number of reviews (e.g., 100)")

    @field_validator('max_year')
    @classmethod
    def validate_year_range(cls, v, info):
        if v is not None and info.data and info.data.get('min_year') is not None:
            if v < info.data['min_year']:
                raise ValueError('max_year must be greater than or equal to min_year')
        return v


class HybridRecommendationRequest(BaseModel):
    """Request model for hybrid recommendations with configurable weights and comprehensive filtering"""
    liked_games: List[str] = Field(..., description="List of game IDs that the user likes")
    limit: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    exclude_owned: bool = Field(default=True, description="Whether to exclude games the user already owns")
    weights: Optional[ModelWeights] = Field(default=None, description="Custom model weights (defaults to balanced)")
    filters: Optional[GameFilters] = Field(default=None, description="Advanced filtering options")

    # Legacy support - keep exclude_franchise at top level for backwards compatibility
    exclude_franchise: Optional[bool] = Field(default=None, description="DEPRECATED: Use filters.exclude_franchise instead")


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
