# Cross-Domain Generalization of Deep Learning Detectors for Rumors and Fake News in Online Social Networks

A fully reproducible, **API-free** study of whether deep learning misinformation
detectors that score near-perfectly in-domain actually transfer across corpora.
Everything runs end to end on a single free Colab GPU with no social-media API keys
and no manual downloads.

## Key finding

High in-domain accuracy is largely an illusion produced by corpus-specific artifacts,
and it does **not** survive domain shift. Transformer pretraining narrows the gap but
does not close it, and transfer is strongly asymmetric.

| Model | McIntire in-domain (Acc / AUC) | LIAR→McIntire (AUC) | McIntire→LIAR (AUC) |
|---|---|---|---|
| LogReg (TF-IDF) | 0.945 / 0.984 | 0.543 | 0.498 |
| BiLSTM | 0.875 / 0.939 | 0.515 | 0.526 |
| DistilBERT | 0.962 / 0.994 | 0.595 | 0.515 |
| RoBERTa | 0.980 / 0.999 | **0.723** | 0.564 |

A model that scores 0.945–0.980 in-domain drops toward chance out-of-domain. RoBERTa
recovers substantial signal for statement→article transfer (AUC 0.723) but the
article→statement direction stays near chance for **every** model. Interpretability shows
the in-domain article score is carried by scraping/era artifacts (`advertisement`,
`print`, `wikileaks`, `november 2016`), not veracity. Calibration error more than doubles
under shift. See `results/results.json` for all numbers.

## Datasets (public, redistributed in full)

- **LIAR** — 12.8k short PolitiFact statements (Wang, 2017).
- **McIntire fake_or_real_news** — ~6.3k full news articles.

Both are downloaded automatically at run time from public GitHub mirrors; no API is used.

## How to run

1. Open `Rumor_FakeNews_CrossDomain.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. **Run all** (~15–25 minutes).
4. Results are written to `results/results.json` and figures under `results/figs/`.

Locally: `pip install -r requirements.txt` then run the notebook with Jupyter.

## Repository structure

```
Rumor_FakeNews_CrossDomain.ipynb   reproducible pipeline (data → models → audit)
results/results.json               serialized metrics used in the paper
figures/                           generalization, asymmetry, calibration figures
paper/IET_Manuscript.docx          manuscript
requirements.txt                   pinned dependencies
LICENSE                            MIT
```

## Method

A controlled ladder of models — TF-IDF linear classifiers (LogReg, Linear SVM, MultinomialNB),
a BiLSTM, and fine-tuned DistilBERT and RoBERTa — evaluated under a strict in-domain vs
cross-domain protocol, with bootstrap confidence intervals, McNemar tests, expected
calibration error with reliability diagrams, and SHAP / linear-coefficient interpretability.
Seed fixed at 42.

## Citation

Gaur, L. (2026). Cross-Domain Generalization of Deep Learning Detectors for Rumors and
Fake News in Online Social Networks: An Auditable and Reproducible Study.
Code and data: https://github.com/LoveleenGaur/rumor-fakenews-crossdomain

## License

Code released under the MIT License. The datasets remain under their original terms.
