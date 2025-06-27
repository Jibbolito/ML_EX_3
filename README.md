# ML Exercise 3: Attribute Inference Attacks on Machine Learning Models

## Topic 3.1.1.2: Attribute Inference (disclosure) from ML Models

This project implements attribute inference attacks on machine learning models using the **German Credit dataset**, following the methodology from Fredrikson et al. (USENIX Security 2014).

## Objective
Test whether trained classification models leak information that allows inferring hidden/missing sensitive attributes from incomplete samples.

## Prerequisites
- Python 3.8+
- Virtual environment recommended
- Internet connection (for downloading German Credit dataset)

## Installation
```bash
# Create and activate virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset: German Credit
- **1000 samples** with financial and personal data
- **Classification target**: Credit approval (good/bad)
- **13 Sensitive attributes tested**: 
  - `checking_status` (checking account status)
  - `credit_history` (credit payment history)
  - `purpose` (loan purpose)
  - `savings_status` (savings account level)
  - `employment` (employment duration/stability)
  - `personal_status` (marital/family status)
  - `other_parties` (other debtors/guarantors)
  - `property_magnitude` (property ownership)
  - `other_payment_plans` (other installment plans)
  - `housing` (housing situation: rent/own/free)
  - `job` (job classification)
  - `own_telephone` (telephone ownership)
  - `foreign_worker` (foreign worker status)

## Attack Method
1. Take incomplete sample X (missing one sensitive attribute like `personal_status`)
2. Try all possible values for the missing attribute 
3. Use model's confidence/probability output to determine which value is most likely correct
4. Test on both training data (X used in training) and validation data (X not used in training)
5. Compare with ground truth to measure attack success

## Models Tested
- **Decision Tree** (prone to overfitting - no depth limit, minimal regularization)
- **Neural Network** (multilayer perceptron - large architecture, no regularization)
- **Random Forest** (ensemble method - many trees, no regularization)
- **Logistic Regression** (well-regularized baseline for comparison)

## Implementation
Based on Fredrikson et al.'s model inversion algorithm A_π that maximizes:
```
Pr[x_t|x_K, y, f] ∝ Σ π_{y,f(x)} * Π p(x_i)
```

Where:
- x_t: target sensitive attribute to infer (e.g., personal_status)
- x_K: known attributes (e.g., credit_history, purpose, employment)
- y: model prediction (credit approval)
- f: trained classification model
- π: model performance statistics
- p: marginal attribute distributions

## Usage

### 1. Validate Implementation
```bash
# Test all components work correctly
python test_implementation.py
```

### 2. Run Full Experiment
```bash
# Execute complete attribute inference attack experiment
python experiments.py
```

### 3. Programmatic Usage
```python
from experiments import AttributeInferenceExperiment

# Create and run experiment
experiment = AttributeInferenceExperiment()
results = experiment.run_full_experiment()

# Access detailed results
vulnerability_analysis = results['vulnerability_analysis']
```

## Output and Results

### Console Output
- **Model Training Summary**: Overfitting gaps for each model type
- **Attack Results**: Training vs validation accuracy for each attribute/model combination
- **Training Advantage**: Difference between training and validation attack success
- **Vulnerability Analysis**: Risk levels and most vulnerable attributes
- **Key Findings**: Best attacks and correlation with overfitting

### Generated Files
- `attack_results_german_credit.png`: Comprehensive visualization of results
- Console logs with detailed metrics and analysis

### Key Metrics
- **Training Advantage**: Positive values indicate training data leakage (expected)
- **Over Baseline**: How much better than random guessing
- **Risk Levels**: HIGH/MEDIUM/LOW/MINIMAL based on training advantage
- **Overfitting Correlation**: Relationship between model overfitting and attack success

## Expected Results
- **Overfitting models** (decision tree, neural network) should show higher training advantages
- **Some attributes** may show negative training advantages (validation performs better)
- **Weak but measurable** information leakage (typically 1-5% training advantage)
- **Attribute-specific patterns** based on correlations in the dataset

## Files Structure
- `attack_implementation.py`: Core Fredrikson model inversion attack algorithm
- `data_preparation.py`: German Credit dataset loading and preprocessing
- `model_training.py`: Train overfitting-prone classification models
- `experiments.py`: Complete experimental pipeline and analysis
- `test_implementation.py`: Comprehensive validation of all components
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## Detailed Implementation Architecture

### Core Classes and Data Structures

#### `ModelInversionAttack` Class (`attack_implementation.py`)

The main attack implementation contains three critical data structures:

**1. `self.marginals` - Marginal Distributions Dictionary**
```python
# Structure: Dict[str, Dict[Any, float]]
# Purpose: Prior probability distributions for all attributes

self.marginals = {
    'own_telephone': {
        'yes': 0.596,    # 59.6% of people own telephones
        'no': 0.404      # 40.4% don't own telephones
    },
    'credit_history': {
        'critical account': 0.293,
        'existing credits paid back duly': 0.361,
        'delay in paying off in the past': 0.088,
        'no credits taken': 0.040,
        'all credits at this bank paid back duly': 0.218
    },
    'duration': {
        '(11.0, 15.0]': 0.12,    # Discretized numerical ranges
        '(15.0, 20.0]': 0.18,
        # ... more bins for numerical features
    }
}
```

**2. `self.performance_stats` - Model Performance Metrics**
```python
# Structure: Dict[str, Any]
# Purpose: Model reliability metrics for π(y, f(x)) calculation

self.performance_stats = {
    'confusion_matrix': [
        [240.0, 15.0],   # True class 0: 240 correct, 15 incorrect
        [45.0, 180.0]    # True class 1: 45 incorrect, 180 correct
    ],
    'class_labels': [0, 1],  # Binary classification labels
    'accuracy': 0.875        # Overall model accuracy (87.5%)
}
```

**3. `self.preprocessing_info` - Data Transformation Metadata**
```python
# Structure: Dict[str, Any]
# Purpose: Convert human-readable attributes to model input format

self.preprocessing_info = {
    'target_column': 'class',
    'categorical_columns': [
        'checking_status', 'credit_history', 'purpose', 
        'savings_status', 'employment', 'personal_status',
        # ... all categorical features
    ],
    'feature_names': [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment',
        # ... exact feature ordering as expected by model
    ],
    'label_encoders': {
        'checking_status': LabelEncoder(['<0', '0<=X<200', '>=200', 'no checking']),
        'own_telephone': LabelEncoder(['no', 'yes']),  # 'yes' → 1, 'no' → 0
        # ... encoders for all categorical features
    },
    'target_encoder': LabelEncoder(['bad', 'good']),  # 'bad' → 0, 'good' → 1
    'scaler': StandardScaler()  # Fitted on training data
}
```

#### Critical Data Flow: `_attributes_to_input()` Method

The attack converts human-readable attributes to model input through this pipeline:

1. **Input**: `{'own_telephone': 'yes', 'credit_history': 'critical account', ...}`
2. **Label Encoding**: `'yes' → 1, 'critical account' → 0`
3. **Feature Ordering**: Arrange according to `feature_names` list (crucial!)
4. **Scaling**: Apply `StandardScaler` to numerical features
5. **Output**: Numpy array compatible with trained model

#### Key Components by File

**`attack_implementation.py`:**
- `ModelInversionAttack`: Main attack orchestrator with Fredrikson's A_π algorithm
- `AttackEvaluator`: Measures attack success and training vs validation comparison
- `calculate_marginal_distributions()`: Computes population statistics
- `calculate_model_performance_stats()`: Analyzes model reliability

**`data_preparation.py`:**
- `DatasetLoader`: Downloads and cleans German Credit dataset from OpenML
- `DataPreprocessor`: Handles label encoding, scaling, and train/val/test splits
- `load_and_prepare_dataset()`: Main pipeline returning processed data + metadata

**`model_training.py`:**
- `ModelTrainer`: Trains overfitting-prone models (Decision Tree, Neural Network, Random Forest)
- Deliberately configured for maximum overfitting to demonstrate information leakage
- Tracks overfitting gaps (training accuracy - validation accuracy)

**`experiments.py`:**
- `AttributeInferenceExperiment`: Complete experimental framework
- Tests each model against each sensitive attribute
- Generates vulnerability analysis and visualizations
- Produces comprehensive privacy risk assessment

## Why German Credit Dataset?
1. **Classification task** with clear binary target (credit approval)
2. **13 categorical sensitive attributes** with limited values to test efficiently
3. **Real-world financial privacy relevance** - personal/financial attribute inference
4. **Known correlations** between personal status, employment, and credit decisions
5. **Models return confidence scores** required for successful attribute inference
6. **Moderate size** (1000 samples) suitable for educational experimentation

## Academic Context
This implementation demonstrates the privacy risks identified by Fredrikson et al. in their seminal USENIX Security 2014 paper. The experiment shows how machine learning models, even when only releasing predictions, can leak sensitive information about individuals' private attributes through model inversion attacks.

## Troubleshooting
- **Dataset download fails**: Check internet connection and OpenML availability
- **Import errors**: Ensure virtual environment is activated and dependencies installed
- **Low attack success**: Expected - real-world attribute inference is challenging
- **Negative training advantages**: Valid finding - indicates complex model behavior