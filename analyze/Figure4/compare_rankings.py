import pandas as pd

# Read files
aiemg_path = r'\..\aiemg.csv'
chembl_path = r'\..\HER2_docked copy.csv'

aiemg = pd.read_csv(aiemg_path)
chembl = pd.read_csv(chembl_path)

print(f'AIEMG molecules: {len(aiemg)}')
print(f'ChEMBL molecules: {len(chembl)}')

# Add source column
aiemg['source'] = 'AIEMG'
chembl['source'] = 'ChEMBL'

# Combine
combined = pd.concat([
    aiemg[['smiles', 'her2', 'source']].rename(columns={'her2': 'docking'}),
    chembl[['smiles', 'egfr', 'source']].rename(columns={'egfr': 'docking'})
], ignore_index=True)

# Sort by docking (low to high)
combined = combined.sort_values('docking', ascending=True).reset_index(drop=True)
combined['rank'] = range(1, len(combined) + 1)

# Save combined ranked data
output_path = r'E:\MCTS\2026.7.10修改意见版本\chembl网页\chembl_with_dock\aiemg_vs_chembl_her2_ranked.csv'
combined.to_csv(output_path, index=False)

# Report AIEMG rankings
aiemg_ranked = combined[combined['source'] == 'AIEMG'].copy()
print(f'\n{"="*60}')
print(f'AIEMG Ranking Report (vs ChEMBL HER2)')
print(f'{"="*60}')
print(f'Total molecules: {len(combined)}')
print(f'AIEMG molecules: {len(aiemg_ranked)}')
print(f'ChEMBL molecules: {len(combined) - len(aiemg_ranked)}')

print(f'\nAIEMG Ranking Statistics:')
print(f'  Best rank: {aiemg_ranked["rank"].min()}')
print(f'  Worst rank: {aiemg_ranked["rank"].max()}')
print(f'  Median rank: {aiemg_ranked["rank"].median():.0f}')
print(f'  Mean rank: {aiemg_ranked["rank"].mean():.1f}')

print(f'\nTop 20 AIEMG molecules:')
top20 = aiemg_ranked.head(20)[['rank', 'docking', 'smiles']]
for _, row in top20.iterrows():
    print(f'  Rank {row["rank"]:4d}: {row["docking"]:.2f} | {row["smiles"][:60]}...')

# Calculate percentile
aiemg_ranked['percentile'] = (1 - aiemg_ranked['rank'] / len(combined)) * 100
print(f'\nPercentile distribution:')
print(f'  Top 1%: {(aiemg_ranked["percentile"] >= 99).sum()} molecules')
print(f'  Top 5%: {(aiemg_ranked["percentile"] >= 95).sum()} molecules')
print(f'  Top 10%: {(aiemg_ranked["percentile"] >= 90).sum()} molecules')
print(f'  Top 25%: {(aiemg_ranked["percentile"] >= 75).sum()} molecules')
print(f'  Top 50%: {(aiemg_ranked["percentile"] >= 50).sum()} molecules')

print(f'\nSaved to: {output_path}')
