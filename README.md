# Machine Learning Regression and Classification Assignment

This project contains two reproducible machine-learning workflows:

1. **House price prediction** with Linear Regression.
2. **Titanic survival prediction** with Logistic Regression.

Both workflows separate training and evaluation data, keep preprocessing inside scikit-learn pipelines, save trained models, and write results to the `reports/` directory.

## Project structure

```text
.
├── data/
│   ├── house_prices/train.csv       # Download separately from Kaggle
│   └── titanic/train.csv            # Download separately from Kaggle
├── models/                          # Generated trained-model files
├── reports/figures/                 # Generated charts
├── src/
│   ├── train_house_prices.py
│   └── train_titanic.py
├── tests/
│   └── test_workflows.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Datasets

Download the training CSV files from Kaggle and place them as follows:

- House Prices: `data/house_prices/train.csv`
- Titanic: `data/titanic/train.csv`

The datasets are intentionally excluded from version control. Review the Kaggle dataset terms before using or sharing them.

## Run the projects

Train the house-price model, calculate R2 and MAE, and create a predicted-versus-actual plot:

```bash
python src/train_house_prices.py
```

Train the Titanic classifier and print accuracy and the classification report:

```bash
python src/train_titanic.py
```

Use `--help` with either script to see options for custom input, output, model, or random-seed paths.

## Test the workflows

```bash
pytest
```

## Expected outputs

After the scripts run, this project produces:

- `models/house_price_linear_regression.joblib`
- `models/titanic_logistic_regression.joblib`
- `reports/house_price_metrics.json`
- `reports/titanic_metrics.json`
- `reports/figures/house_price_predicted_vs_actual.png`

## Notes on evaluation

- **R2** indicates the fraction of price variation explained by the regression model; values nearer to 1 generally indicate a better fit.
- **MAE** is the average prediction error in the original house-price units.
- **Accuracy** is the fraction of Titanic passengers correctly classified as survived or not survived.

## License

Licensed under the MIT License. See [LICENSE](LICENSE).
