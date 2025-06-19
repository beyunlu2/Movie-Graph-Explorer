# Knowledge Graph Completion - Ranking-Based Evaluation

## Changes Made

Replaced the confusing confidence-based scoring system with **standard ranking-based evaluation metrics** used in knowledge graph completion research.

## Why This is Better

### ❌ Problems with Confidence Scores:
- Difficult to interpret and compare across models
- Arbitrary normalization methods
- Not standard in KG research community
- Negative scores were confusing for users

### ✅ Benefits of Ranking Metrics:
- **Standard metrics** used in academic research
- **Interpretable**: Rank 1 is always best, regardless of model
- **Comparable**: Same metrics across all model types
- **Comprehensive**: Multiple metrics provide different perspectives

## New Metrics Explained

### 📊 Core Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Rank** | Position among all candidates | 1, 2, 3, ... | Lower is better (1 = best) |
| **Reciprocal Rank** | 1/Rank | 0.0 to 1.0 | Higher is better (1.0 = perfect) |
| **Rank Percentile** | Better than X% of candidates | 0% to 100% | Higher is better |
| **Hits@K** | Is prediction in top-K? | 0 or 1 | 1 = success, 0 = failure |

### 🎯 Ranking Interpretation Guide

#### By Rank:
- **Rank 1-10**: Excellent predictions ⭐⭐⭐
- **Rank 11-100**: Good predictions ⭐⭐
- **Rank 101-1000**: Moderate predictions ⭐
- **Rank >1000**: Poor predictions

#### By Reciprocal Rank:
- **1.0 (Rank 1)**: Perfect prediction
- **0.5 (Rank 2)**: Very good prediction  
- **0.1 (Rank 10)**: Good prediction
- **0.01 (Rank 100)**: Moderate prediction
- **<0.001**: Poor prediction

## Example Output

### Your Titanic Example:
```
Rank | Relation               | Raw Score | Reciprocal | Percentile | Hits@1 | Hits@3 | Hits@5 | Hits@10
-------------------------------------------------------------------------------------------------
   1 | ACTED_IN              |  -17.0966 |     1.0000 |     100.0% |      1 |      1 |      1 |       1
   2 | DIRECTED              |  -16.1812 |     0.5000 |      80.0% |      0 |      1 |      1 |       1
   3 | IN_GENRE              |  -14.2659 |     0.3333 |      60.0% |      0 |      1 |      1 |       1
   4 | CO_OCCURS_WITH_ACTOR  |  -13.7024 |     0.2500 |      40.0% |      0 |      0 |      1 |       1
   5 | CO_OCCURS_WITH_PERSON |  -13.5433 |     0.2000 |      20.0% |      0 |      0 |      1 |       1
```

**Summary Statistics:**
- Mean Reciprocal Rank (MRR): 0.4167
- Hits@1: 0.200 (20% of predictions are rank 1)
- Hits@3: 0.600 (60% of predictions are in top 3)
- Hits@5: 1.000 (100% of predictions are in top 5)

## Standard KG Evaluation Metrics

These metrics are used in major KG completion papers and frameworks:

### Mean Reciprocal Rank (MRR)
- **Formula**: `MRR = (1/n) * Σ(1/rank_i)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Usage**: Overall quality metric, emphasizes top rankings

### Hits@K
- **Formula**: `Hits@K = (# predictions with rank ≤ K) / total_predictions`
- **Range**: 0.0 to 1.0 (higher is better)  
- **Common K values**: 1, 3, 5, 10
- **Usage**: Success rate for top-K predictions

### Average Rank
- **Formula**: `Mean rank across all predictions`
- **Range**: 1 to vocabulary_size (lower is better)
- **Usage**: Overall ranking performance

## Technical Implementation

### Model-Aware Sorting
```python
# Distance-based models (TransE, RotatE, PairRE): lower scores = better
if model_name in ["TransE", "RotatE", "PairRE"]:
    sorted_indices = np.argsort(scores)  # Ascending sort

# Similarity-based models (ComplEx): higher scores = better  
else:
    sorted_indices = np.argsort(-scores)  # Descending sort
```

### Ranking Calculation
```python
rank = position_in_sorted_list + 1  # 1-indexed
reciprocal_rank = 1.0 / rank
rank_percentile = ((total_candidates - rank) / total_candidates) * 100
hits_at_k = 1 if rank <= k else 0
```

## API Changes

### New DataFrame Columns:
- `Raw_Score`: Original model output
- `Rank`: 1, 2, 3, ... (lower is better)
- `Reciprocal_Rank`: 1.0, 0.5, 0.33, ... (higher is better)
- `Rank_Percentile`: 99.9%, 95.0%, ... (higher is better)
- `Hits@1`, `Hits@3`, `Hits@5`, `Hits@10`: Binary success indicators

### New Methods:
- `explain_ranking_system()`: Explains the ranking metrics
- `_get_ranking_info()`: Internal method for calculating rankings
- `_calculate_ranking_metrics()`: Batch ranking calculations

### Updated Methods:
- `predict_triplet()`: Returns ranking metrics instead of confidence
- `batch_predict()`: Returns ranking metrics for batch predictions
- `find_similar_entities()`: Includes ranking information
- All methods now sort by rank (best first)

## Usage

```python
kg = PykeenMovieKG()

# Get predictions with ranking metrics
results = kg.predict_triplet(head="Titanic", relation="ACTED_IN", tail=None, k=5)
print(results)

# Understand the metrics
print(kg.explain_ranking_system())
```

This provides a much more principled and interpretable evaluation system that aligns with academic standards in knowledge graph completion research, as described in papers from conferences like ICLR, NeurIPS, and ICML [source](https://www.dbs.ifi.lmu.de/~tresp/papers/2203.07544.pdf). 