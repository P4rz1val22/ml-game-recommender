import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

games_df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global games_df
    logger.info("🚀 Starting Game Recommendation API...")

    current_file = Path(__file__)
    project_root = current_file.parent.parent
    data_path = project_root / "data" / "steam_games.json"

    logger.info(f"Current file: {current_file}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Looking for data at: {data_path}")
    logger.info(f"File exists: {data_path.exists()}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    from utils.data_loader import SteamJSONLoader
    loader = SteamJSONLoader(str(data_path))
    games_df = loader.json_to_dataframe()

    logger.info(f"✅ Loaded {len(games_df):,} games")
    yield
    logger.info("🛑 Shutting down Game Recommendation API...")


app = FastAPI(
    title="Game Recommendation API",
    description="ML-powered game recommendations with hidden gem discovery",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "🎮 Game Recommendation API is running!", "games_loaded": len(games_df) if games_df is not None else 0}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-game-recommender",
        "games_loaded": len(games_df) if games_df is not None else 0
    }


@app.get("/api/games/stats")
async def get_games_stats():
    """Get comprehensive dataset statistics"""
    if games_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    total_games = len(games_df)

    high_engagement = len(games_df[games_df['estimated_owners'] != '0 - 20000'])
    low_engagement = len(games_df[games_df['estimated_owners'] == '0 - 20000'])

    ownership_order = ['50000000 - 100000000', '20000000 - 50000000', '10000000 - 20000000', '5000000 - 10000000', '2000000 - 5000000', '1000000 - 2000000', '500000 - 1000000', '200000 - 500000', '100000 - 200000', '50000 - 100000', '20000 - 50000']

    most_owned = []
    for tier in ownership_order:
        tier_games = games_df[games_df['estimated_owners'] == tier]
        if len(tier_games) > 0:
            top_in_tier = tier_games.nlargest(5, 'positive')
            for _, game in top_in_tier.iterrows():
                most_owned.append({
                    "name": game['name'],
                    "owners": tier.replace(' - ', '-')
                })
                if len(most_owned) >= 5:
                    break
        if len(most_owned) >= 5:
            break

    top_genres = games_df['genres'].str.split(',').explode().value_counts().head(5).to_dict()

    genre_quality = {}
    for genre in list(top_genres.keys())[:5]:
        genre_games = games_df[games_df['genres'].str.contains(genre, na=False)]
        reviewed_games = genre_games[(genre_games['positive'] + genre_games['negative']) >= 10]
        if len(reviewed_games) > 0:
            avg_positive_ratio = reviewed_games['positive'] / (reviewed_games['positive'] + reviewed_games['negative'])
            genre_quality[genre] = round(avg_positive_ratio.mean(), 2)

    platforms = {
        "windows": int(games_df['windows'].sum()),
        "mac": int(games_df['mac'].sum()),
        "linux": int(games_df['linux'].sum())
    }

    return {
        "dataset_overview": {
            "total_games": total_games,
            "data_coverage": "2003-2024"
        },
        "engagement": {
            "most_owned_games": most_owned,
            "distribution": {
                "high_engagement": high_engagement,
                "low_engagement": low_engagement
            }
        },
        "content": {
            "top_genres": top_genres,
            "genre_quality": genre_quality,
            "platforms": platforms
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
