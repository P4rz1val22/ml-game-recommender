# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv
from api.recommendations import initialize_recommender, router as recommendations_router
from api.games import router as games_router

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

games_df = None
loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global games_df, loader
    logger.info("🚀 Starting RAWG Game Recommendation API...")

    # Initialize RAWG data loader
    from utils.data_loader import RAWGDatasetLoader
    loader = RAWGDatasetLoader()
    games_df = loader.load_and_process_data()

    logger.info(f"✅ Loaded {len(games_df):,} RAWG games")

    # Initialize recommender with engagement-based popular games
    initialize_recommender(games_df, loader)

    # Initialize search service after recommender is ready
    from api.recommendations import recommender
    from services.game_search import GameSearchService
    import api.games as games_api

    games_api.search_service = GameSearchService(
        recommender.popular_games,
        recommender.game_id_to_index
    )
    logger.info("✅ Game search service ready!")

    yield
    logger.info("🛑 Shutting down RAWG Game Recommendation API...")


app = FastAPI(
    title="RAWG Game Recommendation API",
    description="ML-powered multi-platform game recommendations with engagement-based popularity",
    version="2.0.0",
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
    return {
        "message": "🎮 RAWG Game Recommendation API is running!",
        "games_loaded": len(games_df) if games_df is not None else 0,
        "data_source": "RAWG Video Games Database",
        "platforms": "Multi-platform support"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "rawg-ml-game-recommender",
        "games_loaded": len(games_df) if games_df is not None else 0,
        "data_source": "RAWG"
    }


app.include_router(recommendations_router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(games_router, prefix="/api/games", tags=["games"])

# TODO: Implement RAWG-specific stats endpoint later
# @app.get("/api/games/stats")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
