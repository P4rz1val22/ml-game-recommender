# api/games.py

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from api.models import GameDetail
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Global search service (set in main.py)
search_service = None


@router.get("/search", response_model=List[GameDetail])
async def search_games(name: str, limit: int = 10):
    """Search games by name - returns detailed format"""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    results = search_service.search_games_by_name(name, limit)
    # Remove match_score from API response (internal ranking only)
    clean_results = [{k: v for k, v in game.items() if k != 'match_score'} for game in results]
    return clean_results


@router.get("/{game_id}", response_model=GameDetail)
async def get_game_by_id(game_id: str):
    """Get game details by ID - same format as search"""
    if search_service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    game = search_service.get_game_by_id(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    return game
