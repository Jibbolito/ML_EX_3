#!/usr/bin/env python3
"""
Train models that are prone to overfitting for attribute inference attacks
Following Fredrikson et al.'s recommendation to use neural networks and decision trees
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class OverfittingModels:
    """
    Train models that are intentionally prone to overfitting
    These models are more likely to leak information about training data
    """
    
    @staticmethod
    def train_overfitting_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train decision tree with minimal regularization to encourage overfitting
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with model and training info
        """
        # Intentionally overfitting parameters
        model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=None,          # No depth limit
            min_samples_split=2,     # Minimum samples to split
            min_samples_leaf=1,      # Minimum samples in leaf
            max_features=None,       # Use all features
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate performance
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val)) if X_val is not None else None
        
        return {
            'model': model,
            'type': 'decision_tree',
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'overfitting_gap': train_acc - val_acc if val_acc else None,
            'description': 'Overfitting Decision Tree (no regularization)'
        }
    
    @staticmethod
    def train_overfitting_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train neural network with parameters that encourage overfitting
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with model and training info
        """
        n_features = X_train.shape[1]
        
        # Overfitting parameters: large network, no regularization
        model = MLPClassifier(
            hidden_layer_sizes=(n_features * 2, n_features, n_features // 2),  # Large network
            activation='relu',
            solver='adam',
            alpha=0.0,              # No L2 regularization
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=False,   # No early stopping
            validation_fraction=0.0, # Don't use validation for early stopping
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate performance
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val)) if X_val is not None else None
        
        return {
            'model': model,
            'type': 'neural_network',
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'overfitting_gap': train_acc - val_acc if val_acc else None,
            'description': 'Overfitting Neural Network (large, no regularization)'
        }
    
    @staticmethod
    def train_overfitting_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train random forest with parameters that encourage overfitting
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with model and training info
        """
        model = RandomForestClassifier(
            n_estimators=200,        # Many trees
            max_depth=None,          # No depth limit
            min_samples_split=2,     # Minimum samples to split
            min_samples_leaf=1,      # Minimum samples in leaf
            max_features='sqrt',     # Feature sampling
            bootstrap=False,         # Use all samples (more overfitting)
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate performance
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val)) if X_val is not None else None
        
        return {
            'model': model,
            'type': 'random_forest',
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'overfitting_gap': train_acc - val_acc if val_acc else None,
            'description': 'Overfitting Random Forest (many trees, no regularization)'
        }
    
    @staticmethod
    def train_baseline_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                                         X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train well-regularized logistic regression as baseline
        Should be less prone to overfitting
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with model and training info
        """
        model = LogisticRegression(
            C=1.0,                   # Moderate regularization
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate performance
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val)) if X_val is not None else None
        
        return {
            'model': model,
            'type': 'logistic_regression',
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'overfitting_gap': train_acc - val_acc if val_acc else None,
            'description': 'Baseline Logistic Regression (regularized)'
        }


class ModelTrainer:
    """
    Train multiple models and compare their overfitting characteristics
    """
    
    def __init__(self):
        self.models = {}
        
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train all model types and compare overfitting behavior
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of trained models with performance metrics
        """
        print("Training models for attribute inference attack...")
        
        # Train overfitting-prone models
        self.models['decision_tree'] = OverfittingModels.train_overfitting_decision_tree(
            X_train, y_train, X_val, y_val
        )
        
        self.models['neural_network'] = OverfittingModels.train_overfitting_neural_network(
            X_train, y_train, X_val, y_val
        )
        
        self.models['random_forest'] = OverfittingModels.train_overfitting_random_forest(
            X_train, y_train, X_val, y_val
        )
        
        # Train baseline model
        self.models['logistic_regression'] = OverfittingModels.train_baseline_logistic_regression(
            X_train, y_train, X_val, y_val
        )
        
        # Print summary
        self._print_training_summary()
        
        return self.models
    
    def _print_training_summary(self):
        """Print summary of model training results"""
        print("\n" + "="*70)
        print("MODEL TRAINING SUMMARY")
        print("="*70)
        print(f"{'Model Type':<20} {'Train Acc':<12} {'Val Acc':<12} {'Overfitting Gap':<15}")
        print("-"*70)
        
        for model_name, model_info in self.models.items():
            train_acc = model_info['train_accuracy']
            val_acc = model_info['val_accuracy'] or 0
            gap = model_info['overfitting_gap'] or 0
            
            print(f"{model_name:<20} {train_acc:<12.3f} {val_acc:<12.3f} {gap:<15.3f}")
        
        print("-"*70)
        
        # Identify most overfitting model
        max_gap = max([m['overfitting_gap'] for m in self.models.values() if m['overfitting_gap']])
        most_overfitting = [name for name, model in self.models.items() 
                          if model['overfitting_gap'] == max_gap][0]
        
        print(f"Most overfitting model: {most_overfitting} (gap: {max_gap:.3f})")
        print("="*70 + "\n")
    
    def get_model(self, model_type: str):
        """Get trained model by type"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Available: {list(self.models.keys())}")
        return self.models[model_type]['model']
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get full model information by type"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Available: {list(self.models.keys())}")
        return self.models[model_type]
    
    def rank_by_overfitting(self) -> List[Tuple[str, float]]:
        """
        Rank models by overfitting gap (highest first)
        
        Returns:
            List of (model_name, overfitting_gap) tuples, sorted by gap descending
        """
        ranking = [(name, info['overfitting_gap']) for name, info in self.models.items() 
                  if info['overfitting_gap'] is not None]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking


def train_models_for_attack(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> ModelTrainer:
    """
    Convenience function to train all models for attribute inference attack
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        ModelTrainer instance with all trained models
    """
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_val, y_val)
    return trainer


if __name__ == "__main__":
    # Test model training with synthetic data
    from data_preparation import load_and_prepare_dataset
    
    print("Testing model training...")
    
    # Load German Credit dataset
    data = load_and_prepare_dataset()
    
    # Train models
    trainer = train_models_for_attack(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val']
    )
    
    # Show overfitting ranking
    print("Overfitting ranking:")
    for model_name, gap in trainer.rank_by_overfitting():
        print(f"  {model_name}: {gap:.3f}")