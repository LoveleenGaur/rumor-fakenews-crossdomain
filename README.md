# Cross-Corpus Generalization and Calibration of Deep Learning Fake-News Detectors

A reproducible, **API-free**, **multi-seed** study of how far fake-news detectors generalize
across corpora. Runs end to end on a free Colab T4 GPU with no social-media API keys.
Code and data: https://github.com/LoveleenGaur/rumor-fakenews-crossdomain

## Key finding

In-domain near-saturation coexists with a **genre-structured** generalization gap. Transfer
succeeds within the news genre and, with pretraining, from statements to news, but transfer
**into** the short-statement corpus (LIAR) stays at chance for every model. Vocabulary overlap
predicts which transfers succeed, and interpretability shows in-domain success on news leans on
scraping/source artifacts (the token `com` is a top fake cue in *both* news corpora).

## In-domain (RoBERTa, 3-seed mean)

| Corpus | Accuracy | ROC-AUC |
|---|---|---|
| LIAR (statements) | 0.618 | 0.657 |
| McIntire (news) | 0.989 | 0.999 |
| ISOT (news) | 0.980 | 0.999 |

## Cross-corpus transfer, ROC-AUC (RoBERTa)

| Train \ Test | LIAR | McIntire | ISOT |
|---|---|---|---|
| **LIAR** | (in-domain) | 0.765 | 0.935 |
| **McIntire** | 0.527 | (in-domain) | 0.943 |
| **ISOT** | 0.514 | 0.792 | (in-domain) |

Transfer into LIAR is at chance (AUC 0.51–0.53) for every model; news-to-news transfer is strong.
Full metrics with per-cell standard deviations, calibration, interpretability, bootstrap CIs, and
McNemar tests are in `results/results.json`.

## Datasets (public, no API)

- **LIAR** (Wang, 2017) — short PolitiFact statements.
- **McIntire** — full news articles.
- **ISOT** (Ahmed, Traore & Saad, 2017) — Reuters vs flagged-site articles (via the GonzaloA mirror).

Cross-corpus de-duplication removes any article shared between corpora; each training set is capped
at 5,000 for a controlled comparison.

## How to run

Open `Rumor_FakeNews_CrossDomain.ipynb` in Colab, set a T4 GPU, Run all (~60–90 min for 3 seeds;
set `SEEDS` to 5 entries for a 5-seed run). Results are written to `results/results.json`.

## Repository structure

```
Rumor_FakeNews_CrossDomain.ipynb   reproducible 3-corpus, multi-seed pipeline
results/results.json               full serialized metrics
figures/                           transfer matrix, calibration, and Supplementary S1
paper/IET_Manuscript.docx          manuscript
requirements.txt / LICENSE
```

## Citation

Raman, R. & Gaur, L. Cross-Corpus Generalization and Calibration of Deep Learning Fake-News
Detectors: A Reproducible, Multi-Seed Audit (2026).

## License

Code under the MIT License; datasets retain their original terms.
