# Quick franchise data analysis
from src.utils.data_loader import RAWGDatasetLoader

loader = RAWGDatasetLoader()
df = loader.load_and_process_data()

# Show franchise examples
franchise_examples = df[df['name'].str.contains('Half-Life|Call of Duty|Grand Theft|Assassin|Far Cry', case=False, na=False)]['name'].head(15)
print("Franchise examples:")
for name in franchise_examples.tolist():
    print(f"  {name}")
