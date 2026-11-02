# How Many BERT Dimensions Does Sentiment Classification Need?

Code for our STAT 236 course project (Harvard University, Fall 2026).

We extract frozen `bert-base-uncased` embeddings (768 dimensions) for Amazon product reviews and test
how much sentiment-classification performance is kept when only k dimensions are used. The k dimensions
are chosen in three ways: at random, by L1-regularized logistic regression, or by PCA.

## What's in this repo

```
stat236_embedding.ipynb   # the full pipeline in one notebook
README.md
```

The notebook runs these steps in order:

| Step | What it does |
|---|---|
| 0 · Settings | All configurable options: sample size, model, k values, seeds, run mode |
| 1 · Data | Samples a balanced train/test set from [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) and cleans the text |
| 2 · Embeddings | Encodes each review with frozen BERT (masked mean pooling) and saves the vectors |
| EDA | Plots review lengths, the structure of the embedding coordinates, and PCA explained variance |
| 3 · Experiments | Compares the full embedding with random, L1-selected and PCA-reduced representations of size k, all using the same L2 logistic regression classifier |
| 4 · Plots | Plots accuracy and macro-F1 against k |

## How to run

The notebook is meant for **Google Colab** with a GPU (*Runtime → Change runtime type → T4 GPU*).

1. Open `stat236_embedding.ipynb` in Colab.
2. **Quick check:** keep the defaults `EMB_LIMIT = 200` and `QUICK = True` in Step 0, then *Run all* (a few minutes).
3. **Full run:** set `EMB_LIMIT = None` and `QUICK = False`, then *Run all* (about 15–25 minutes).

Each step saves its output and is skipped on re-runs if that output already exists. The notebook
creates these folders:

```
data/          # sampled reviews (parquet)
embeddings/    # BERT embeddings (.npy)
results/       # experiment tables, figures, EDA outputs
```

`data/` and `embeddings/` are large and are not tracked by git.

To run locally instead of on Colab:

```bash
pip install torch transformers datasets pyarrow scikit-learn pandas matplotlib
```

Without a GPU, the embedding step will be slow.