from pydantic import BaseModel, Field
from typing import List, Optional


class RecommendationRequest(BaseModel):
    """Request model for game recommendations"""
    liked_games: List[str] = Field(
        ...,
        description="List of game IDs for games the user has liked",
        example=["58175"]  # Half-Life 2, Left 4 Dead 2, Portal
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
    """Individual game recommendation - RAWG schema"""
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


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    recommendations: List[GameRecommendation]
    total_found: int
    algorithm: str = "content_based_rawg_engagement"
    processing_time_ms: Optional[float] = None


class GameInfo(BaseModel):
    """Detailed game information - RAWG schema"""
    game_id: str
    name: str
    description: str
    genres: str
    rating: float
    ratings_count: int
    added_to_library: int
    platforms: str
