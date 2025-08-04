import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path


class SteamJSONLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.raw_data = {}
        self.df = None

    def load_json(self) -> Dict:
        """Load raw JSON data"""
        print(f"Loading JSON from {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        print(f"✅ Loaded {len(self.raw_data):,} games from JSON")
        return self.raw_data

    def json_to_dataframe(self) -> pd.DataFrame:
        """Convert JSON to clean pandas DataFrame"""
        if not self.raw_data:
            self.load_json()

        games_list = []

        for app_id, game in self.raw_data.items():
            game_data = {
                'app_id': app_id,
                'name': game.get('name', ''),
                'release_date': game.get('release_date', ''),
                'estimated_owners': game.get('estimated_owners', ''),
                'peak_ccu': game.get('peak_ccu', 0),
                'required_age': game.get('required_age', 0),
                'price': game.get('price', 0.0),
                'dlc_count': game.get('dlc_count', 0),

                'detailed_description': game.get('detailed_description', ''),
                'short_description': game.get('short_description', ''),
                'supported_languages': ','.join(game.get('supported_languages', [])),

                'positive': game.get('positive', 0),
                'negative': game.get('negative', 0),
                'metacritic_score': game.get('metacritic_score', 0),
                'user_score': game.get('user_score', 0),
                'recommendations': game.get('recommendations', 0),

                'genres': ','.join(game.get('genres', [])),
                'categories': ','.join(game.get('categories', [])),
                'tags': ','.join([str(tag) for tag in game.get('tags', [])]),
                'developers': ','.join(game.get('developers', [])),
                'publishers': ','.join(game.get('publishers', [])),

                'windows': game.get('windows', False),
                'mac': game.get('mac', False),
                'linux': game.get('linux', False),

                'achievements': game.get('achievements', 0),
                'average_playtime_forever': game.get('average_playtime_forever', 0),
                'median_playtime_forever': game.get('median_playtime_forever', 0),
            }

            games_list.append(game_data)

        self.df = pd.DataFrame(games_list)
        print(f"✅ Created DataFrame with shape: {self.df.shape}")
        return self.df

    def get_clean_data_summary(self):
        """Get comprehensive data summary"""
        if self.df is None:
            self.json_to_dataframe()

        owners_dist = self.df['estimated_owners'].value_counts()
        indie_games = len(self.df[self.df['estimated_owners'] == '0 - 20000'])
        popular_games = len(self.df[self.df['estimated_owners'] != '0 - 20000'])

        return {
            'total_games': len(self.df),
            'indie_games': indie_games,
            'popular_games': popular_games,
            'avg_positive_rating': self.df['positive'].mean(),
            'avg_metacritic': self.df[self.df['metacritic_score'] > 0]['metacritic_score'].mean(),
            'top_genres': self.df['genres'].str.split(',').explode().value_counts().head(10),
            'missing_descriptions': self.df['detailed_description'].isna().sum(),
            'price_distribution': self.df['price'].describe()
        }


# Example usage with proper path handling
if __name__ == "__main__":
    # Get absolute path to data file
    current_file = Path(__file__)  # src/utils/data_loader.py
    project_root = current_file.parent.parent.parent  # ml-game-recommender/
    data_path = project_root / "data" / "steam_games.json"

    print(f"Current file: {current_file}")
    print(f"Project root: {project_root}")
    print(f"Data path: {data_path}")
    print(f"File exists: {data_path.exists()}")

    # Load data
    loader = SteamJSONLoader(str(data_path))
    df = loader.json_to_dataframe()
    summary = loader.get_clean_data_summary()

    print("\n📊 Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


    def print_full_games_sample(df, n=5):
        """Print first n games with all columns for inspection"""
        for i in range(min(n, len(df))):
            game = df.iloc[i]
            print(f"\n{'=' * 60}")
            print(f"🎮 GAME {i + 1}: {game['name']}")
            print(f"{'=' * 60}")

            for column in df.columns:
                value = game[column]
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                else:
                    display_value = value

                print(f"{column:25s}: {display_value}")


    print_full_games_sample(df, 5)

    print(f"\n{'=' * 60}")
    print("📋 DATAFRAME INFO:")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
