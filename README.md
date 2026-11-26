# SynthLens: Detecting Synthetic Product Reviews

A hybrid approach combining NLP vectorization and stylistic feature fusion using scikit-learn pipelines.

**Author:** Swayam Shakya
**Co-author:** Bibek Khadka

## Folder contents

```
SynthLens.ipynb        the full project notebook (already executed, with all outputs; short code comments explain
                       each step, and the full discussion is in the project report)
requirements.txt       the libraries needed to run the notebook
data/                  the CSV files used (see "Data" below)
figures/               every figure from the notebook as a PNG, ready for the report
```

## How to run

1. Install the requirements: `pip install -r requirements.txt`
2. Start Jupyter from this folder: `jupyter notebook` (or `jupyter lab`)
3. Open `SynthLens.ipynb` and choose **Run > Run All Cells**.

Run the notebook from the top. From Section 17 on, some cells delete objects from earlier sections to save memory, so re-running one of those cells on its own can fail; restart the kernel and run all cells instead.

The notebook took between about 85 and 105 minutes on a 2-core cloud machine, mostly in the grid searches (Section 10), Part C and Section 17. Section 17 is the most memory-hungry part (the Python process peaked at about 5.7 GB), so 8 GB of RAM or more is recommended.

`N_JOBS` in the first code cell sets how many models are trained in parallel. It is set to 2, the value used for every reported result. Each parallel job holds its own copy of the data, so on a computer with 16 GB of RAM or more you can raise it to 4 to finish sooner; the results stay the same.

**Library versions.** Use a recent Python 3 installation. The minimum library versions are listed in `requirements.txt`; the notebook was tested without errors on both those minimum versions and current releases. With older library releases, the linear models give the same results to three or four decimal places, but the random forest (Section 10) can differ by about a point, because its implementation has changed over time.

## Data

| File(s) | Dataset | Used in | Original source |
|---|---|---|---|
| `fake_reviews_dataset.csv` | Fake Reviews Dataset: 40,432 Amazon reviews, half human, half GPT-2 generated | Part A | Salminen et al. (2022), https://doi.org/10.1016/j.jretconser.2021.102771 |
| `train_v2_drcat_02.csv` | DAIGT V2 Train Dataset: 44,868 essays, human and modern LLM | Part B | https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset |
| `train_essays.csv` | LLM - Detect AI Generated Text competition essays (used only to check overlap) | Part B | https://www.kaggle.com/competitions/llm-detect-ai-generated-text |
| `mage_yelp_full_train.csv`, `mage_yelp_full_valid.csv`, `mage_yelp_full_test.csv` | MAGE benchmark, Yelp rows of the official splits: reviews by humans and 27 LLMs | Part C | Y. Li et al. (2024), https://huggingface.co/datasets/yaful/MAGE |
| `test_ood_set_gpt.csv`, `test_ood_set_gpt_para.csv` | MAGE GPT-4 test sets (the notebook uses the IMDb movie reviews, plain and paraphrased) | Part C | Y. Li et al. (2024), https://huggingface.co/datasets/yaful/MAGE |
| `aigen_train.csv`, `aigen_val.csv`, `aigen_test.csv` | AiGen-FoodReview: 20,144 NYC restaurant reviews, half by Yelp Elite users, half by GPT-4-Turbo | Part C | Gambetti & Han (2024), https://github.com/iamalegambetti/AiGen-FoodReview |

The `mage_yelp_full_*.csv` files are the Yelp rows of MAGE's official `train.csv`, `valid.csv` and `test.csv` (the full training file is about 400 MB). If they are missing, the notebook rebuilds them from the official files placed in `data/mage/`, or downloads those from Hugging Face.

Any other missing file is downloaded automatically: the MAGE and AiGen files from their official releases, and the Kaggle files from identical public GitHub copies
(https://github.com/SayamAlt/Fake-Reviews-Detection and https://github.com/pinskyrobin/LLM---Detect-AI-Generated-Text).

## Headline results

All settings were chosen by cross-validation or on validation splits, and each test set was scored once.

| Benchmark | Baseline hybrid | SynthLens+ |
|---|---|---|
| GPT-2 Amazon reviews (Part A, held-out test) | 96.0% | **96.9%** |
| The same reviews after post-editing (hardened model) | 89.2% | **93.7%** |
| Essays from unseen prompts (Part B, DAIGT V2) | 96.2% | **97.0%** |
| Yelp reviews by 27 LLMs (Part C, MAGE) | 74.3% | **77.3%** |
| GPT-4-Turbo restaurant reviews (Part C, AiGen) | 99.7% | 99.7% |
| GPT-4 movie reviews, unseen domain (trained on AiGen) | 95.5% | 95.5% |
| The same movie reviews after GPT-4 paraphrasing | 89.0% | 88.3% |

Section 18 of the notebook collects every headline number in one table. The discussion of these results, the four kinds of leakage examined and the limitations are in the project report.

## Checking your own review

Section 19 of the notebook turns the two strongest detectors into a simple checker. After running the notebook, copy a review from any online store, paste it between the triple quotes in the cell marked `19.3 Paste a review here`, and run that cell. SynthLens reports whether the review looks human-written or synthetic, with the answer of each detector:

* **Detector 1** (older generators, GPT-2 style): the hardened SynthLens+ model, refit on all Part A reviews.
* **Detector 2** (modern generators, GPT-4 style): the SynthLens+ model trained on AiGen-FoodReview.

Several reviews can be checked at once with `check_reviews([...])`, which returns a table. The checker is a screening aid: very short reviews, text from newer models and AI-polished human reviews are the main sources of mistakes.
