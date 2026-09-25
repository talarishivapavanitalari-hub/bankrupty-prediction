# Corporate Bankruptcy Prediction

A comparative study of **Decision Tree** and **Gradient Boosting** classifiers for predicting corporate bankruptcy using financial ratio data.

## 📌 Overview
This project builds and compares two machine learning models to classify companies as bankrupt or non-bankrupt based on 95 financial indicators. It highlights a key real-world ML lesson: a simpler, interpretable model (Decision Tree) can outperform a more complex ensemble method (Gradient Boosting) on Recall and F1-Score when data is heavily imbalanced.

## 📊 Dataset
- **Source**: [Taiwanese Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) (UCI / Kaggle)
- **Size**: 6,819 companies, 95 financial ratio features
- **Target**: `Bankrupt?` (0 = No, 1 = Yes)
- **Class distribution**: 96.8% non-bankrupt, 3.2% bankrupt (highly imbalanced)

> Note: A comparable public dataset for Indian corporate bankruptcy was not available at the time of this project. Extending this work to Indian financial data is noted as future scope.

## 🛠️ Tech Stack
- Python 3.10
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Git & GitHub

## 📈 Results

| Metric    | Decision Tree | Gradient Boosting |
|-----------|:-------------:|:------------------:|
| Accuracy  | 96.85%        | 96.92%             |
| Precision | 51.43%        | 53.85%             |
| Recall    | 40.91%        | 31.82%             |
| F1-Score  | 45.57%        | 40.00%             |

**Key finding**: Decision Tree achieved better Recall and F1-Score, making it more effective at catching actual bankruptcy cases in this imbalanced dataset.

## 🔑 Top Predictive Features
1. Net Value Growth Rate
2. Interest-bearing Debt Interest Rate
3. Borrowing Dependency
4. Continuous Interest Rate (after tax)
5. ROA(B) before interest & depreciation, after tax

## 📁 Project Structure
```
bankrupty-prediction/
├── data/                   # dataset (excluded from Git via .gitignore)
├── src/
│   └── bankrupty_pipeline.py
├── reports/
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   └── Bankruptcy_Prediction_Report.pdf
├── notebooks/
├── .gitignore
└── README.md
```

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python src/bankrupty_pipeline.py
```

## 🔮 Future Scope
- Apply SMOTE / class-weighting to handle imbalance
- Extend to Indian corporate financial data (NCLT/IBC records)
- Compare against XGBoost, LightGBM, Random Forest
