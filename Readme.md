# Bloom's Taxonomy Question Classifier – Model Experiments

This repository holds the experiments used to classify exam questions into the six levels of **Bloom's Taxonomy** (BT1–BT6). It contains three models, the dataset they were trained on, and a keyword mapping used to tag each question with a subject and Course Outcome (CO).

The TensorFlow/Keras model in `main.ipynb` is the one selected for the project.

---

## Repository structure

```
blooms_models/
├── data/
│   ├── blooms_taxonomy_dataset.csv   # 14,847 labelled questions
│   └── co_mapping.json               # subject + Course Outcome (CO) keyword mapping
└── notebook/
    ├── main.ipynb                    # Model 3 – TF-IDF + Keras MLP (final / selected)
    ├── test_2.ipynb                  # Model 1 – TF-IDF + Keras MLP (early run, ~10.5k rows)
    └── test3.ipynb                   # Model 2 – PyTorch TextCNN
```

## Dataset

`data/blooms_taxonomy_dataset.csv` has two columns: `Questions` (text) and `Category` (label).

| Label | Bloom's level | Samples |
| ----- | ------------- | ------- |
| BT1   | Remember      | 3,956   |
| BT2   | Understand    | 3,133   |
| BT3   | Apply         | 2,760   |
| BT4   | Analyze       | 2,877   |
| BT5   | Evaluate      | 1,051   |
| BT6   | Create        | 1,070   |

The classes are imbalanced (BT5 and BT6 are roughly a quarter the size of BT1), which matters for the model comparison below.

All models use the same preprocessing (lowercase, strip punctuation) and the same split: 80/20 train/test, `stratify=labels`, `random_state=42`.

`data/co_mapping.json` maps keywords to 4 subjects (Data Structures and Algorithms, Discrete Structures and Graph Theory, Database Management Systems, Quantitative Analysis) and to their Course Outcomes (CO1–CO6, or CO1–CO4 for Quantitative Analysis). It is used in `main.ipynb` by `analyze_question()`.

---

## The three models at a glance

|                         | **Model 1** `test_2.ipynb`                                              | **Model 2** `test3.ipynb`                                                                                                 | **Model 3** `main.ipynb`                        |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Framework**           | TensorFlow / Keras                                                      | PyTorch                                                                                                                   | TensorFlow / Keras                              |
| **Architecture**        | Dense MLP                                                               | 1-D TextCNN                                                                                                               | Dense MLP (same as Model 1)                     |
| **Text representation** | TF-IDF, 2,000 features, 1–3-grams                                       | Learned word embeddings (dim 100, vocab 10,718, max length 50)                                                            | TF-IDF, 2,000 features, 1–3-grams               |
| **Layers**              | Dense 128 → BN → Dropout 0.6 → Dense 64 → BN → Dropout 0.6 → Softmax(6) | Embedding → 3 parallel Conv1D (kernel 2/3/4, 100 filters each) → global max-pool → concat (300) → Dropout 0.7 → Linear(6) | Same as Model 1                                 |
| **Parameters**          | ~266 K                                                                  | ~1.16 M                                                                                                                   | ~266 K                                          |
| **Regularisation**      | L2 (0.01), BatchNorm, Dropout 0.6, early stopping                       | Dropout 0.7, weight decay 1e-4, StepLR schedule                                                                           | Same as Model 1                                 |
| **Class imbalance**     | `class_weight='balanced'`                                               | None                                                                                                                      | `class_weight='balanced'`                       |
| **Optimiser**           | Adam, lr 5e-4                                                           | Adam, lr 5e-4, weight decay 1e-4                                                                                          | Adam, lr 5e-4                                   |
| **Training loop**       | `model.fit()` with `EarlyStopping` (patience 3)                         | Hand-written train/test loops, 10 epochs                                                                                  | `model.fit()` with `EarlyStopping` (patience 3) |
| **Dataset size**        | ~10,475 rows (8,380 train)                                              | 14,847 rows (11,877 train / 2,970 test)                                                                                   | 14,847 rows (11,877 train / 2,970 test)         |
| **Test accuracy**       | **80.10 %**                                                             | **83.74 %**                                                                                                               | **86.06 %**                                     |
| **Extras**              | Sample prediction helper                                                | Training/validation loss plot                                                                                             | Subject + CO mapping, saved model artifacts     |

### Model 1 – Keras MLP on TF-IDF (`test_2.ipynb`)

The first baseline. Questions are turned into a 2,000-dimensional TF-IDF vector (uni-, bi- and tri-grams) and passed through a small, heavily regularised feed-forward network. It was trained on the earlier ~10.5k-row version of the dataset and reached 80.10 %. It is kept as the reference point for how much the data increase helped.

### Model 2 – PyTorch TextCNN (`test3.ipynb`)

A different approach: instead of TF-IDF, each word is mapped to an index and passed through a trainable embedding layer. Three parallel convolutions (kernel sizes 2, 3, 4) act as bigram/trigram/4-gram detectors, followed by global max pooling. This model can learn word order and its own embeddings, but it is about 4× larger than the MLP, starts from random embeddings on a modest dataset, and was trained without class weighting. It reached a best of 83.74 % after 10 epochs (accuracy was still creeping up slowly).

### Model 3 – Keras MLP on TF-IDF, larger dataset (`main.ipynb`)

The same architecture and hyperparameters as Model 1, trained on the expanded 14,847-row dataset. It is the best performer at 86.06 %. This notebook also contains the full inference pipeline: `analyze_question()` returns the predicted Bloom level plus the matching subject and CO, and the model, TF-IDF vectorizer and label encoder are saved to `bloom_model.h5`, `vectorizer.pkl` and `label_encoder.pkl`.

> **Models 1 and 3 differ only in data size.** The jump from 80.10 % to 86.06 % comes from adding data (10k → 14k), not from an architecture change.

---

## Why TensorFlow (Keras) over PyTorch

TensorFlow/Keras was chosen for the final model for these reasons:

1. **Best measured accuracy.** On the same dataset and the same train/test split, the Keras model scored 86.06 % versus 83.74 % for the PyTorch model.
2. **Simpler training and less custom code.** Keras handles training, validation, early stopping and class weighting in a few lines (`fit`, `EarlyStopping`, `class_weight`). The PyTorch version needs a custom `Dataset`, `DataLoader`s, a vocabulary builder, manual padding, and hand-written train/test loops. Less custom code means fewer places for bugs.
3. **Class imbalance is handled directly.** `class_weight='balanced'` is a one-argument change in Keras. This matters here because BT5 and BT6 are under-represented.
4. **Small model, small data.** For short exam questions where Bloom's level is largely signalled by keywords ("define", "compare", "design"), TF-IDF n-grams plus a small MLP are a strong fit. A TextCNN's learned embeddings add ~900K parameters without a measurable gain on this dataset.
5. **Straightforward packaging for the app.** The trained model and its preprocessing objects are saved together (`.h5` + vectorizer + label encoder), so the app can load them and call `analyze_question()` without re-implementing the pipeline.

### Caveats on the comparison

The two frameworks were not compared on identical terms, so the accuracy gap should be read as indicative rather than conclusive:

- The models use different input representations (TF-IDF vs. learned embeddings), so the comparison is as much "MLP + TF-IDF vs. TextCNN" as "TensorFlow vs. PyTorch".
- The PyTorch model had no class weighting, a fixed 10 epochs and no early stopping.
- In both Keras notebooks, the test set is also used as the validation set for early stopping. The reported accuracy is therefore slightly optimistic; a separate validation split would give a cleaner estimate.

A like-for-like test (the same TextCNN in Keras, or the same TF-IDF MLP in PyTorch) would be needed to isolate the effect of the framework itself.

---

## Getting started

**Requirements:** Python 3.12, Jupyter, and the packages below.

```bash
pip install numpy pandas scikit-learn tensorflow nltk matplotlib jupyter
# only needed for test3.ipynb
pip install torch
```

**Run the experiments** (from the `notebook/` folder, since the notebooks use relative paths to `../data/`):

```bash
cd notebook
jupyter notebook
```

Open `main.ipynb` for the final model. Running its last cell writes `bloom_model.h5`, `vectorizer.pkl` and `label_encoder.pkl` into the `notebook/` folder.

**Example usage (after running `main.ipynb`):**

```python
analyze_question("Explain the properties of a good estimator")
# returns a dict with: original_question, bloom_level, subject, co_id
```

`test3.ipynb` selects the Apple `mps` device when available and falls back to CPU otherwise.

## Notes and next steps

- `main.ipynb` imports `nltk`, `WordNetLemmatizer` and `PCA` but does not use them yet.
- Consider adding a held-out validation split, per-class precision/recall and a confusion matrix, since overall accuracy can hide weak performance on BT5/BT6.
- Consider adding a `requirements.txt` and a `.gitignore` (the repo currently contains `.DS_Store` files).
