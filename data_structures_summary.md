# Model Inversion Attack Data Structures - Complete Analysis

This document provides the exact structure and contents of the three key data structures used in the `ModelInversionAttack` class for the German Credit dataset.

## Summary of Analysis

I have analyzed the code implementation and provided detailed documentation of:

1. **`self.marginals`** - Marginal probability distributions for all attributes
2. **`self.performance_stats`** - Model performance metrics including confusion matrix
3. **`self.preprocessing_info`** - Preprocessing metadata for data transformation

## Documentation Files (Not Required for Functionality)

**Note**: The following files are documentation/analysis files created to explain the internal data structures. They are NOT functionally required for the attribute inference attack implementation and can be safely deleted if desired.

### 1. `data_structures_analysis.py` - Structural Analysis
**Purpose**: Comprehensive structural analysis of all three data structures
- Shows the exact types, nested structure, and usage patterns
- Explains how each structure is created and used in the attack algorithm
- Provides theoretical understanding of the data format

**Content Preview**:
```python
def analyze_marginals_structure():
    """Analysis of self.marginals structure based on calculate_marginal_distributions()"""
    # Detailed breakdown of Dictionary[str, Dictionary[Any, float]] structure
    # Explains categorical vs numerical attribute handling
    # Shows how probabilities are calculated and normalized

def analyze_performance_stats_structure():
    """Analysis of self.performance_stats structure"""
    # Breakdown of confusion matrix format
    # Explains class_labels and accuracy metrics
    # Shows how π(y, f(x)) is calculated from confusion matrix

def analyze_preprocessing_info_structure():
    """Analysis of self.preprocessing_info structure"""
    # Details of feature_names ordering (critical for attacks)
    # Label encoder mappings for categorical features
    # Scaler information for numerical features
```

### 2. `concrete_examples.py` - Real-World Examples
**Purpose**: Real-world examples with actual German Credit dataset values
- Shows what the data structures look like with concrete data
- Demonstrates step-by-step attack execution using all three structures
- Provides practical understanding with actual values

**Content Preview**:
```python
def show_concrete_marginals_example():
    """Shows actual marginals with real German Credit values"""
    # Example: 'checking_status': {'A11': 0.274, 'A12': 0.269, ...}
    # Example: 'age': {'(19.0, 25.2]': 0.167, '(25.2, 31.4]': 0.183, ...}

def show_concrete_performance_stats_example():
    """Shows actual confusion matrix and accuracy values"""
    # Example: [[129, 71], [38, 162]] for binary classification
    # Shows how to interpret True/False Positives/Negatives

def show_attack_example():
    """Complete end-to-end attack example"""
    # Shows how to infer 'checking_status' given other attributes
    # Step-by-step likelihood calculation using all three structures
```

### 3. `examine_data_structures.py` - Runtime Inspection Script
**Purpose**: Runtime examination script that would show actual runtime values when executed
- Requires dependencies to be installed and data to be loaded
- Provides detailed structure printing with controlled depth
- Interactive exploration of the data structures

**Content Preview**:
```python
def examine_marginals(train_df, categorical_cols):
    """Load actual data and examine marginal distributions structure"""
    
def examine_performance_stats(model, X_test, y_test):
    """Train model and examine actual performance statistics"""
    
def examine_preprocessing_info(preprocessing_info):
    """Show detailed preprocessing information structure"""
```

## Deletion Instructions

These documentation files can be safely removed without affecting functionality:

```bash
# Safe to delete - documentation only
rm data_structures_analysis.py
rm concrete_examples.py  
rm examine_data_structures.py
```

**Core functional files** (keep these):
- `attack_implementation.py` - Core Fredrikson algorithm
- `data_preparation.py` - Dataset loading and preprocessing  
- `model_training.py` - Train overfitting models
- `experiments.py` - Main experimental pipeline
- `test_implementation.py` - Validation suite
- `requirements.txt` - Dependencies
- `README.md` - User documentation

## Key Findings

### 1. Marginals Structure (`self.marginals`)
```python
Type: Dict[str, Dict[Any, float]]

Structure:
{
    'attribute_name': {
        'value1': probability1,
        'value2': probability2,
        ...
    },
    ...
}

Example:
{
    'checking_status': {
        'A11': 0.274,  # "< 0 DM" 
        'A12': 0.269,  # "0 <= ... < 200 DM"
        'A13': 0.063,  # ">= 200 DM"
        'A14': 0.394   # "no checking account"
    }
}
```

### 2. Performance Stats Structure (`self.performance_stats`)
```python
Type: Dict[str, Any]

Structure:
{
    'confusion_matrix': [[int, int, ...], ...],  # 2D list
    'class_labels': [label1, label2, ...],       # list  
    'accuracy': float                            # overall accuracy
}

Example:
{
    'confusion_matrix': [
        [129, 71],   # True class 0: 129 correct, 71 wrong
        [38, 162]    # True class 1: 38 wrong, 162 correct  
    ],
    'class_labels': [0, 1],
    'accuracy': 0.7275
}
```

### 3. Preprocessing Info Structure (`self.preprocessing_info`)
```python
Type: Dict[str, Any]

Structure:
{
    'target_column': str,
    'categorical_columns': List[str],
    'feature_names': List[str],              # CRITICAL: defines feature order
    'label_encoders': Dict[str, LabelEncoder],
    'target_encoder': LabelEncoder,
    'scaler': StandardScaler                 # optional
}

Key Usage:
- feature_names: defines exact order for model input vectors
- label_encoders: convert categorical strings to integers  
- scaler: normalize numerical features
```

## How They Work Together

The three structures implement **Fredrikson's A_π algorithm**:

1. **Marginals** provide **prior probabilities** P(attribute=value)
2. **Performance stats** provide **model reliability** π(y, f(x)) 
3. **Preprocessing info** provides **data transformation** to query the model

### Attack Algorithm
```
For each possible value v of target attribute:
  1. Create complete_attributes = known_attrs + {target: v}
  2. Convert to model input using preprocessing_info
  3. Get model prediction f(x) 
  4. Calculate performance_score using performance_stats
  5. Calculate prior_score using marginals
  6. likelihood = performance_score × prior_score

Return value with highest likelihood
```

## Implementation Details

- **Marginals**: Created by `calculate_marginal_distributions()`
  - Categorical: normalized value counts
  - Numerical: discretized bins with probabilities
  - All probabilities sum to 1.0 per attribute

- **Performance Stats**: Created by `calculate_model_performance_stats()`
  - Confusion matrix enables P(true|predicted) calculations
  - Used in `_get_performance_score()` method

- **Preprocessing Info**: Created by `DataPreprocessor.prepare_for_modeling()`
  - Critical for `_attributes_to_input()` method
  - Ensures attack queries match training data format

This analysis shows how the three data structures work together to enable successful model inversion attacks by combining population statistics, model behavior analysis, and proper data transformation.