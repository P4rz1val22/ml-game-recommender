# data_loader.py

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from datasets import load_dataset


class RAWGDatasetLoader:
    def __init__(self, cache_dir: str = 'cache'):
        self.raw_dataset = None
        self.df = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache file paths
        self.processed_cache = self.cache_dir / "rawg_processed_games.pkl"

    def load_dataset(self) -> pd.DataFrame:
        """Load RAWG dataset from Hugging Face - trying direct parquet download first"""
        print("🎮 Loading RAWG dataset from Hugging Face...")

        try:
            # Try direct parquet download (often more reliable)
            from huggingface_hub import hf_hub_download
            print("📥 Attempting direct parquet download...")

            file_path = hf_hub_download(
                repo_id="atalaydenknalbant/rawg-games-dataset",
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset"
            )

            print(f"✅ Downloaded parquet file to: {file_path}")
            self.df = pd.read_parquet(file_path)
            print(f"✅ Loaded {len(self.df):,} games directly from parquet")

        except Exception as e:
            print(f"⚠️ Direct parquet failed: {e}")
            print("🔄 Falling back to datasets library...")

            # Fallback to datasets library with force redownload
            self.raw_dataset = load_dataset(
                "atalaydenknalbant/rawg-games-dataset",
                download_mode="force_redownload"
            )
            self.df = self.raw_dataset['train'].to_pandas()
            print(f"✅ Loaded {len(self.df):,} games via datasets library")

        print(f"✅ DataFrame shape: {self.df.shape}")
        return self.df

    def load_and_process_data(self) -> pd.DataFrame:
        """Load and process data with caching to avoid 4-minute reloads"""

        # Try to load from cache first
        if self.processed_cache.exists():
            print(f"🚀 Loading processed data from cache: {self.processed_cache}")
            try:
                self.df = pd.read_pickle(self.processed_cache)
                print(f"✅ Loaded {len(self.df):,} processed games from cache in <1 second!")
                return self.df
            except Exception as e:
                print(f"⚠️ Cache loading failed: {e}")
                print("🔄 Falling back to full download...")

        # If no cache, do full download and processing
        print("📥 No cache found, loading and processing data (this will take ~4 minutes)...")
        raw_df = self.load_dataset()
        processed_df = self.clean_and_process_data()

        # Save to cache for next time
        try:
            print(f"💾 Saving processed data to cache...")
            processed_df.to_pickle(self.processed_cache)
            print(f"✅ Cache saved! Next run will be <1 second")
        except Exception as e:
            print(f"⚠️ Cache saving failed: {e}")

    def clean_and_process_data(self) -> pd.DataFrame:
        """Clean and process the RAWG data for use with recommendation engine"""
        if self.df is None:
            self.load_dataset()

        print("🧹 Cleaning and processing RAWG data...")

        # Create cleaned version
        processed_df = self.df.copy()

        # Handle missing values for critical fields
        processed_df['description_raw'] = processed_df['description_raw'].fillna('')
        processed_df['description'] = processed_df['description'].fillna('')
        processed_df['name'] = processed_df['name'].fillna('')
        processed_df['genres'] = processed_df['genres'].fillna('')
        processed_df['tags'] = processed_df['tags'].fillna('')
        processed_df['platforms'] = processed_df['platforms'].fillna('')
        processed_df['developers'] = processed_df['developers'].fillna('')
        processed_df['publishers'] = processed_df['publishers'].fillna('')

        # Convert numeric fields, handling any issues
        processed_df['rating'] = pd.to_numeric(processed_df['rating'], errors='coerce').fillna(0.0)
        processed_df['metacritic'] = pd.to_numeric(processed_df['metacritic'], errors='coerce').fillna(0)
        processed_df['ratings_count'] = pd.to_numeric(processed_df['ratings_count'], errors='coerce').fillna(0)
        processed_df['playtime'] = pd.to_numeric(processed_df['playtime'], errors='coerce').fillna(0)
        processed_df['added'] = pd.to_numeric(processed_df['added'], errors='coerce').fillna(0)

        # Clean release dates
        processed_df['released'] = pd.to_datetime(processed_df['released'], errors='coerce')

        # Ensure ID is string for consistency
        processed_df['id'] = processed_df['id'].astype(str)

        # Filter out games with no meaningful description
        initial_count = len(processed_df)
        processed_df = processed_df[
            (processed_df['description_raw'].str.len() > 50) |  # Has detailed description OR
            (processed_df['description'].str.len() > 20)  # Has short description
            ].copy()

        filtered_count = initial_count - len(processed_df)
        if filtered_count > 0:
            print(f"🗑️ Filtered out {filtered_count:,} games with insufficient descriptions")

        print(f"✅ Processed {len(processed_df):,} games with good descriptions")
        return processed_df
        return processed_df

    def get_popular_games_filter(self, min_added: int = 100, min_rating: float = 2.0) -> pd.DataFrame:
        """Filter to popular games based on RAWG metrics (replaces Steam's estimated_owners logic)"""
        if self.df is None:
            self.load_dataset()

        # RAWG equivalent of "popular" games:
        # - At least min_added users have added to library
        # - Minimum rating threshold
        # - Has meaningful ratings (not just 1-2 ratings)
        popular_games = self.df[
            (self.df['added'] >= min_added) &
            (self.df['rating'] >= min_rating) &
            (self.df['ratings_count'] >= 10)  # At least 10 ratings for reliability
            ].copy()

        print(f"🔥 Popular games filter: {len(popular_games):,} games")
        print(f"   - Min added to library: {min_added:,}")
        print(f"   - Min rating: {min_rating}")
        print(f"   - Min ratings count: 10")

        return popular_games

    def get_engagement_based_popular_games(self, debug: bool = False) -> pd.DataFrame:
        """
        Create an engagement-focused popularity filter based ONLY on player activity
        No rating filters - popularity != quality
        """
        if self.df is None:
            self.load_dataset()

        from datetime import datetime

        if debug:
            print("🚀 CREATING ENGAGEMENT-BASED POPULARITY FILTER")
            print("=" * 60)
            print("Philosophy: Popularity = Player Engagement, NOT Rating Quality")

        # Ensure we have clean release dates
        df_clean = self.df.copy()
        df_clean['released'] = pd.to_datetime(df_clean['released'], errors='coerce')

        # Calculate years since release
        current_year = datetime.now().year
        df_clean['years_since_release'] = current_year - df_clean['released'].dt.year

        # Define recency categories with ENGAGEMENT-ONLY thresholds
        def get_engagement_thresholds(years_since_release):
            """Return (min_added, min_ratings_count) based on recency - NO RATING FILTER"""

            if pd.isna(years_since_release):
                # Unknown release date - moderate engagement threshold
                return 50, 5

            if years_since_release <= 1:
                # Very recent games (2024+) - very low barrier
                return 5, 3  # Just 5 people tried it, 3 bothered to rate
            elif years_since_release <= 2:
                # Recent games (2022-2023) - low barrier
                return 15, 5
            elif years_since_release <= 5:
                # Modern games (2019-2021) - moderate barrier
                return 30, 8
            elif years_since_release <= 10:
                # Established games (2014-2018) - standard barrier
                return 75, 10
            else:
                # Classic games (pre-2014) - higher engagement for staying power
                return 100, 15

        # Apply thresholds based on recency
        popular_games = []

        recency_stats = {
            'very_recent': {'count': 0, 'threshold': '≥5 added, ≥3 reviews (any rating)'},
            'recent': {'count': 0, 'threshold': '≥15 added, ≥5 reviews (any rating)'},
            'modern': {'count': 0, 'threshold': '≥30 added, ≥8 reviews (any rating)'},
            'established': {'count': 0, 'threshold': '≥75 added, ≥10 reviews (any rating)'},
            'classic': {'count': 0, 'threshold': '≥100 added, ≥15 reviews (any rating)'},
            'unknown_date': {'count': 0, 'threshold': '≥50 added, ≥5 reviews (any rating)'}
        }

        for idx, game in df_clean.iterrows():
            years = game['years_since_release']
            min_added, min_reviews = get_engagement_thresholds(years)

            # Check if game meets its recency-adjusted ENGAGEMENT threshold
            # NO RATING FILTER - we don't care if it's good, just if people played it
            if (game['added'] >= min_added and
                    game['ratings_count'] >= min_reviews):

                popular_games.append(idx)

                # Track stats by category
                if pd.isna(years):
                    recency_stats['unknown_date']['count'] += 1
                elif years <= 1:
                    recency_stats['very_recent']['count'] += 1
                elif years <= 2:
                    recency_stats['recent']['count'] += 1
                elif years <= 5:
                    recency_stats['modern']['count'] += 1
                elif years <= 10:
                    recency_stats['established']['count'] += 1
                else:
                    recency_stats['classic']['count'] += 1

        popular_df = df_clean.loc[popular_games].copy()

        if debug:
            print(f"\n📊 ENGAGEMENT-BASED RESULTS:")
            print(f"Total popular games: {len(popular_df):,}")
            print(f"\nBreakdown by recency category (NO RATING FILTERS):")
            for category, stats in recency_stats.items():
                if stats['count'] > 0:
                    print(f"  {category.replace('_', ' ').title()}: {stats['count']:,} games ({stats['threshold']})")

            # Compare with rating-based approach
            rating_based = df_clean[
                (df_clean['added'] >= 100) &
                (df_clean['rating'] >= 3.0) &
                (df_clean['ratings_count'] >= 10)
                ]

            print(f"\n📈 COMPARISON WITH RATING-BASED:")
            print(f"Engagement-only: {len(popular_df):,} games")
            print(f"Rating-based: {len(rating_based):,} games")
            print(f"Difference: +{len(popular_df) - len(rating_based):,} games")

            # Show average rating of engagement-based games
            avg_rating = popular_df['rating'].mean()
            print(f"\nAverage rating of engagement-popular games: {avg_rating:.2f}")
            print("(This includes 'bad' but popular games - which is the point!)")

        return popular_df
        """
        Create a recency-aware popularity filter that adjusts thresholds based on release date
        Recent games get lower barriers to entry
        """
        if self.df is None:
            self.load_dataset()

        from datetime import datetime

        if debug:
            print("🕒 CREATING RECENCY-AWARE POPULARITY FILTER")
            print("=" * 60)

        # Ensure we have clean release dates
        df_clean = self.df.copy()
        df_clean['released'] = pd.to_datetime(df_clean['released'], errors='coerce')

        # Calculate years since release
        current_year = datetime.now().year
        df_clean['years_since_release'] = current_year - df_clean['released'].dt.year

        # Define recency categories with different thresholds
        def get_popularity_thresholds(years_since_release):
            """Return (min_added, min_rating, min_ratings_count) based on recency"""

            if pd.isna(years_since_release):
                # Unknown release date - use moderate threshold
                return 75, 3.0, 8

            if years_since_release <= 1:
                # Very recent games (2024+) - very low barrier
                return 10, 2.8, 3
            elif years_since_release <= 2:
                # Recent games (2022-2023) - low barrier
                return 25, 2.9, 5
            elif years_since_release <= 5:
                # Modern games (2019-2021) - moderate barrier
                return 50, 3.0, 8
            elif years_since_release <= 10:
                # Established games (2014-2018) - standard barrier
                return 100, 3.1, 10
            else:
                # Classic games (pre-2014) - higher barrier (proven staying power)
                return 150, 3.2, 15

        # Apply thresholds based on recency
        popular_games = []

        recency_stats = {
            'very_recent': {'count': 0, 'threshold': '≥10 added, ≥2.8 rating'},
            'recent': {'count': 0, 'threshold': '≥25 added, ≥2.9 rating'},
            'modern': {'count': 0, 'threshold': '≥50 added, ≥3.0 rating'},
            'established': {'count': 0, 'threshold': '≥100 added, ≥3.1 rating'},
            'classic': {'count': 0, 'threshold': '≥150 added, ≥3.2 rating'},
            'unknown_date': {'count': 0, 'threshold': '≥75 added, ≥3.0 rating'}
        }

        for idx, game in df_clean.iterrows():
            years = game['years_since_release']
            min_added, min_rating, min_reviews = get_popularity_thresholds(years)

            # Check if game meets its recency-adjusted threshold
            if (game['added'] >= min_added and
                    game['rating'] >= min_rating and
                    game['ratings_count'] >= min_reviews):

                popular_games.append(idx)

                # Track stats by category
                if pd.isna(years):
                    recency_stats['unknown_date']['count'] += 1
                elif years <= 1:
                    recency_stats['very_recent']['count'] += 1
                elif years <= 2:
                    recency_stats['recent']['count'] += 1
                elif years <= 5:
                    recency_stats['modern']['count'] += 1
                elif years <= 10:
                    recency_stats['established']['count'] += 1
                else:
                    recency_stats['classic']['count'] += 1

        popular_df = df_clean.loc[popular_games].copy()

        if debug:
            print(f"\n📊 RECENCY-AWARE RESULTS:")
            print(f"Total popular games: {len(popular_df):,}")
            print(f"\nBreakdown by recency category:")
            for category, stats in recency_stats.items():
                if stats['count'] > 0:
                    print(f"  {category.replace('_', ' ').title()}: {stats['count']:,} games ({stats['threshold']})")

            # Compare with simple threshold
            simple_threshold = df_clean[
                (df_clean['added'] >= 100) &
                (df_clean['rating'] >= 3.0) &
                (df_clean['ratings_count'] >= 10)
                ]

            print(f"\n📈 COMPARISON:")
            print(f"Recency-aware: {len(popular_df):,} games")
            print(f"Simple threshold: {len(simple_threshold):,} games")
            print(f"Difference: +{len(popular_df) - len(simple_threshold):,} games")

        return popular_df

    def filter_by_platform(self, platform_keywords: List[str]) -> pd.DataFrame:
        """Filter games by platform keywords"""
        if self.df is None:
            self.load_dataset()

        # Create boolean mask for platform filtering
        platform_pattern = '|'.join(platform_keywords)
        platform_mask = self.df['platforms'].str.contains(platform_pattern, case=False, na=False)
        filtered_df = self.df[platform_mask].copy()

        print(f"✅ Platform filter ({platform_keywords}): {len(filtered_df):,} games")
        return filtered_df

    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data quality and distribution summary"""
        if self.df is None:
            self.load_dataset()

        # Description quality
        has_detailed_desc = (self.df['description_raw'].str.len() > 50).sum()
        has_short_desc = (self.df['description'].str.len() > 10).sum()
        avg_desc_length = self.df['description_raw'].str.len().mean()

        # Platform distribution (top 10)
        all_platforms = []
        for platforms_str in self.df['platforms'].dropna():
            if platforms_str and str(platforms_str) != 'nan':
                platforms = [p.strip() for p in str(platforms_str).split('|')]
                all_platforms.extend(platforms)
        platform_counts = pd.Series(all_platforms).value_counts().head(10)

        # Genre distribution (top 10)
        all_genres = []
        for genres_str in self.df['genres'].dropna():
            if genres_str and str(genres_str) != 'nan':
                genres = [g.strip() for g in str(genres_str).split(',')]
                all_genres.extend(genres)
        genre_counts = pd.Series(all_genres).value_counts().head(10)

        # Rating analysis
        rating_stats = self.df['rating'].describe()
        high_rated = (self.df['rating'] >= 4.0).sum()

        # Popularity analysis
        popular_by_added = (self.df['added'] >= 100).sum()
        very_popular = (self.df['added'] >= 1000).sum()

        return {
            'total_games': len(self.df),
            'description_quality': {
                'has_detailed_description': has_detailed_desc,
                'has_short_description': has_short_desc,
                'avg_description_length': round(avg_desc_length, 1)
            },
            'top_platforms': platform_counts.to_dict(),
            'top_genres': genre_counts.to_dict(),
            'rating_distribution': {
                'average_rating': round(rating_stats['mean'], 2),
                'min_rating': round(rating_stats['min'], 2),
                'max_rating': round(rating_stats['max'], 2),
                'high_rated_games_4plus': high_rated
            },
            'popularity_metrics': {
                'games_added_100plus': popular_by_added,
                'games_added_1000plus': very_popular,
                'avg_added_to_library': round(self.df['added'].mean(), 1)
            },
            'date_range': {
                'earliest_release': str(self.df['released'].min())[:10] if pd.notna(self.df['released'].min()) else 'N/A',
                'latest_release': str(self.df['released'].max())[:10] if pd.notna(self.df['released'].max()) else 'N/A'
            }
        }


def print_sample_games(df: pd.DataFrame, n: int = 3, sort_by: str = 'added', recent_years: int = 5) -> None:
    """Print sample games for inspection"""
    print(f"\n{'=' * 80}")
    print(f"🎮 SAMPLE GAMES ({n} of {len(df):,})")
    print(f"{'=' * 80}")

    for i in range(min(n, len(df))):
        game = df.iloc[i]
        print(f"\n🎯 GAME {i + 1}: {game.get('name', 'Unknown')}")
        print(f"{'─' * 60}")
        print(f"ID: {game.get('id', 'N/A')}")
        print(f"Rating: {game.get('rating', 'N/A')} ({game.get('ratings_count', 0):,} ratings)")
        print(f"Added to libraries: {game.get('added', 0):,}")
        print(f"Platforms: {str(game.get('platforms', 'N/A'))[:60]}...")
        print(f"Genres: {str(game.get('genres', 'N/A'))[:60]}...")
        print(f"Metacritic: {game.get('metacritic', 'N/A')}")

        # Show description preview
        desc = game.get('description_raw', game.get('description', ''))
        if desc and len(str(desc)) > 0:
            preview = str(desc)[:150] + "..." if len(str(desc)) > 150 else str(desc)
            print(f"Description: {preview}")
        else:
            print("Description: [No description available]")


def print_top_games_analysis(df: pd.DataFrame) -> None:
    """Show top 5 games from 2025 only"""

    print(f"\n🔥 TOP GAMES ANALYSIS")
    print("=" * 80)

    # Top games from 2025 only
    print(f"\nTop games from 2025")
    games_2025 = df[df['released'].dt.year == 2026].copy()
    if len(games_2025) > 0:
        top_2025 = games_2025.nlargest(10, 'added')
        for i, (idx, game) in enumerate(top_2025.iterrows()):
            print(f"  {i + 1}. {game['name']}")
    else:
        print("  No games found from 2025")


# Example usage and testing
if __name__ == "__main__":
    print("🎮 RAWG Dataset Loader")
    print("=" * 50)

    # Initialize and load with caching
    loader = RAWGDatasetLoader()
    df = loader.load_and_process_data()  # This will use cache on subsequent runs!

    # Get summary (using cached processed data)
    # summary = loader.get_data_summary()

    # print("\n📊 DATASET SUMMARY:")
    # print("=" * 50)
    # for key, value in summary.items():
    #     if isinstance(value, dict):
    #         print(f"{key}:")
    #         for subkey, subvalue in value.items():
    #             print(f"  {subkey}: {subvalue}")
    #     else:
    #         print(f"{key}: {value}")

    # Test popular games filter
    # print(f"\n🔥 TESTING POPULAR GAMES FILTER:")
    # print("=" * 50)
    # popular_df = loader.get_popular_games_filter(min_added=100, min_rating=3.0)

    # Show meaningful sample games instead of random ones
    # print_sample_games(popular_df, n=3, sort_by='added', recent_years=5)

    # Show comprehensive analysis
    # print_top_games_analysis(popular_df)

    print(f"\n✅ RAWG Loader Ready!")
    print(f"✅ Total games: {len(df):,}")
    print(f"✅ Next run will be much faster thanks to caching!")
