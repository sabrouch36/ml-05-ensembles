# ml-05-ensembles

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Ensemble machine learning project comparing Random Forest and Gradient
> Boosting for red wine quality classification.

## Project Description

This project applies ensemble machine learning methods to classify red wine
quality from physicochemical measurements.

The original numeric quality score was converted into three categories:

- **Low:** quality scores 3–4
- **Medium:** quality scores 5–6
- **High:** quality scores 7–8

The custom project compares:

- Random Forest with 200 trees and `max_depth=10`
- Gradient Boosting with 100 estimators

Model performance is evaluated using accuracy, weighted F1, macro F1,
train-test generalization gaps, per-class metrics, confusion matrices, and
feature importance.

## Custom Project Notebook

The fully executed Phase 5 notebook contains the complete analysis, code,
outputs, tables, and charts:

[Open the executed wine-quality notebook](notebooks/project05/ensemble-sabri.ipynb)

Additional notebooks:

- [Original ensemble example](notebooks/ml_05_ensembles.ipynb)
- [Sabri's Phase 4 modification](notebooks/ml_05_sabri.ipynb)

## Dataset

The project uses the Red Wine Quality Dataset from the UCI Machine Learning
Repository.

The original data contained:

- 1,599 records
- 11 physicochemical input features
- one numeric quality score
- no missing values
- 240 exact duplicate rows

After duplicate removal, 1,359 unique records remained.

The cleaned target distribution was strongly imbalanced:

| Quality category | Count | Percentage |
| --- | ---: | ---: |
| Low | 63 | 4.6% |
| Medium | 1,112 | 81.8% |
| High | 184 | 13.5% |

![Red wine quality category distribution](./docs/images/phase5_class_distribution.png)

## Model Results

The majority-class baseline test accuracy was `0.8162`.

| Model | Test Accuracy | Test Weighted F1 | Test Macro F1 | Accuracy Gap |
|---|---:|---:|---:|---:|
| Random Forest | 0.8529 | 0.8241 | 0.4953 | 0.1167 |
| Gradient Boosting | 0.8456 | 0.8232 | 0.5020 | 0.1241 |

Random Forest produced the strongest overall result, with the highest test
accuracy, the highest weighted F1 score, and the smaller accuracy
generalization gap.

Gradient Boosting achieved a slightly higher macro F1 score and identified more
High-quality wines.

![Ensemble model performance comparison](./docs/images/phase5_model_comparison.png)

## Key Findings

Although both models exceeded the majority-class baseline, the improvement was
limited. Random Forest improved test accuracy by approximately 3.7 percentage
points.

Neither model correctly identified any of the 13 Low-quality wines in the test
set. All were classified as Medium. This demonstrates why overall accuracy can
be misleading when the target is imbalanced.

![Random Forest confusion matrix](./docs/images/phase5_confusion_matrix_random_forest.png)

The most influential features across both models were:

1. alcohol
2. volatile acidity
3. sulphates
4. density
5. total sulfur dioxide

![Feature importance by ensemble model](./docs/images/phase5_feature_importance.png)

## Project Setup

Clone the repository and enter the project folder:

```shell
git clone https://github.com/sabrouch36/ml-05-ensembles
cd ml-05-ensembles
```

Create and synchronize the project environment:

```shell
uv python pin 3.14
uv sync --extra dev --extra docs
```

Open the repository in VS Code:

```shell
code .
```

Select the project's `.venv` Python kernel before running the notebooks.

## Run the Project

Open and run the custom notebook in VS Code:

```text
notebooks/project05/ensemble-sabri.ipynb
```

To execute the notebook from the terminal:

```shell
uv run python -m nbconvert --to notebook --execute --inplace notebooks/project05/ensemble-sabri.ipynb
```

Run the original example module:

```shell
uv run python -m mlstudio.app_case
```

Build the documentation site:

```shell
uv run python -m zensical build
```

Run the project quality checks:

```shell
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
```

## Project Structure

```text
ml-05-ensembles/
├── data/raw/
│   └── winequality-red.csv
├── docs/
│   ├── images/
│   └── index.md
├── notebooks/
│   ├── ml_05_ensembles.ipynb
│   ├── ml_05_sabri.ipynb
│   └── project05/
│       └── ensemble-sabri.ipynb
├── src/mlstudio/
├── pyproject.toml
├── zensical.toml
└── README.md
```

## Documentation

- [Project analytics narrative](docs/index.md)
- [Hosted GitHub Pages documentation](https://sabrouch36.github.io/ml-05-ensembles/)
- [Project instructions](docs/project-instructions.md)
- [Glossary](docs/glossary.md)
- [API documentation](docs/api.md)

## Limitations and Future Work

The primary limitation is the severe target imbalance, especially the small
Low-quality class.

Potential improvements include:

- collecting more Low-quality observations
- applying class weighting or resampling
- using stratified cross-validation
- tuning ensemble hyperparameters
- optimizing macro F1 rather than accuracy
- evaluating alternative category boundaries
- using permutation importance or SHAP values

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
