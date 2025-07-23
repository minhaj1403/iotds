# K-EmoPhone Stress Prediction Project

This repository contains code for predicting users' self-reported stress using extracted features from the K-EmoPhone dataset. The project implements various machine learning approaches including traditional ML models and deep learning methods specifically designed for tabular data.

## Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config/
│   └── config.py               # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Data loading and preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── extraction.py       # Feature extraction and categorization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evxgb.py           # Enhanced XGBoost Classifier
│   │   ├── tabnet_model.py    # TabNet implementation
│   │   └── ensemble.py        # Ensemble methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── cross_validation.py # Cross-validation utilities
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── lasso.py           # LASSO-based feature selection
│   │   ├── shap_selection.py  # SHAP-based feature selection
│   │   └── rf_selection.py    # Random Forest feature selection
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
├── scripts/
│   ├── run_assignment1.py     # Feature combination experiments
│   ├── run_assignment2.py     # Feature selection experiments
│   ├── run_assignment3.py     # Hyperparameter tuning
│   ├── run_assignment4.py     # Deep learning models
│   └── run_assignment5.py     # Ensemble methods
├── results/
│   └── .gitkeep               # Results directory placeholder
└── K_EmoPhone_20210753_MiniProject_ipynb.ipynb  # Original notebook
```

## Features

- **Multiple Feature Categories**: Time, DSC, Current sensor, ESM, Yesterday/Today features, Sleep, PIF
- **Feature Selection Methods**: LASSO, SHAP, Random Forest-based selection
- **Models**: Enhanced XGBoost, TabNet, Ensemble methods
- **Evaluation**: Cross-validation with proper group splitting to avoid data leakage
- **Hyperparameter Tuning**: Using Hyperopt for optimal model configuration

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd iotds
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the extracted features dataset from [this link](https://drive.google.com/file/d/1HcyFvzWEzO21osyP5E8VpVmHROX1ew7q/view?usp=sharing) and place it in the project directory.

## Usage

### Running Individual Assignments

```bash
# Assignment 1: Feature combination experiments
python scripts/run_assignment1.py

# Assignment 2: Feature selection experiments
python scripts/run_assignment2.py

# Assignment 3: Hyperparameter tuning
python scripts/run_assignment3.py

# Assignment 4: Deep learning models
python scripts/run_assignment4.py

# Assignment 5: Ensemble methods
python scripts/run_assignment5.py
```

### Using the Library

```python
from src.data.loader import load_kemophone_data
from src.features.extraction import reorder_and_split_features
from src.models.evxgb import EvXGBClassifier
from src.evaluation.cross_validation import perform_cross_validation

# Load data
X, y, groups, t, datetimes = load_kemophone_data('/path/to/features_stress_fixed_K-EmoPhone.pkl')

# Extract features
features = reorder_and_split_features(X, y, groups, datetimes)

# Train model
estimator = EvXGBClassifier(random_state=42)
results = perform_cross_validation(
    features["feat_baseline"], 
    features["y"], 
    features["groups"], 
    estimator
)
```

## Configuration

Edit `config/config.py` to modify:
- Random seed
- Cross-validation settings
- Model parameters
- File paths

## Results

Results from each assignment are saved in the `results/` directory with descriptive filenames indicating the experiment configuration.

## Contributors

This project is based on material from TAs at IC Lab, KAIST, including Panyu Zhang, Soowon Kang, and Woohyeok Choi.

## License

This work is licensed under CC BY-SA 4.0.
