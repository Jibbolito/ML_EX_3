#!/usr/bin/env python3
"""
Implementation of Fredrikson-style Model Inversion Attack for Attribute Inference
Based on "Privacy in Pharmacogenetics" (USENIX Security 2014)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.stats import norm
import itertools
from collections import defaultdict


class ModelInversionAttack:
    """
    Implements the A_π algorithm from Fredrikson et al. for attribute inference
    """
    
    def __init__(self, model, marginal_distributions: Dict, performance_stats: Dict = None,
                 preprocessing_info: Dict = None):
        """
        Initialize the model inversion attack
        
        Args:
            model: Trained ML model with predict_proba method
            marginal_distributions: Dict mapping attribute names to their marginal distributions
            performance_stats: Model performance statistics (confusion matrix, error distribution)
            preprocessing_info: Information about how data was preprocessed for training
        """
        self.model = model
        self.marginals = marginal_distributions
        self.performance_stats = performance_stats or {}
        self.preprocessing_info = preprocessing_info or {}
        
    def infer_attribute(self, known_attributes: Dict, target_attribute: str, 
                       target_class: int = None) -> Tuple[Any, float, Dict]:
        """
        Infer the value of a target attribute given known attributes
        
        Args:
            known_attributes: Dict of known attribute values {attr_name: value}
            target_attribute: Name of attribute to infer
            target_class: Known model prediction (if available)
            
        Returns:
            Tuple of (inferred_value, confidence, all_scores)
        """
        # Get possible values for target attribute
        if target_attribute not in self.marginals:
            raise ValueError(f"No marginal distribution available for {target_attribute}")
            
        possible_values = list(self.marginals[target_attribute].keys())
        
        # Calculate scores for each possible value
        scores = {}
        
        for candidate_value in possible_values:
            # Create complete feature vector
            complete_attributes = known_attributes.copy()
            complete_attributes[target_attribute] = candidate_value
            
            # Calculate likelihood score
            score = self._calculate_likelihood(complete_attributes, target_class)
            scores[candidate_value] = score
            
        # Return value with highest score
        best_value = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_value]
        
        return best_value, confidence, scores
    
    def _calculate_likelihood(self, complete_attributes: Dict, target_class: int = None) -> float:
        """
        Calculate likelihood score for a complete attribute vector
        Following Fredrikson's A_π algorithm
        
        Args:
            complete_attributes: Complete feature vector
            target_class: Known model prediction
            
        Returns:
            Likelihood score
        """
        # Convert attributes to model input format
        X = self._attributes_to_input(complete_attributes)
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            pred_probs = self.model.predict_proba(X.reshape(1, -1))[0]
            model_prediction = np.argmax(pred_probs)
            model_confidence = pred_probs[model_prediction]
        else:
            model_prediction = self.model.predict(X.reshape(1, -1))[0]
            model_confidence = 1.0  # Assume perfect confidence for deterministic models
            
        # Calculate π(y, f(x)) - model performance component
        if target_class is not None:
            performance_score = self._get_performance_score(target_class, model_prediction)
        else:
            performance_score = model_confidence
            
        # Calculate prior probability - product of marginals
        prior_score = 1.0
        for attr_name, attr_value in complete_attributes.items():
            if attr_name in self.marginals:
                prior_score *= self.marginals[attr_name].get(attr_value, 1e-10)
                
        # Combine scores (following Fredrikson's formula)
        likelihood = performance_score * prior_score
        
        return likelihood
    
    def _get_performance_score(self, true_class: int, predicted_class: int) -> float:
        """
        Get performance score π(y, y') from model statistics
        
        Args:
            true_class: True class label
            predicted_class: Model prediction
            
        Returns:
            Performance score
        """
        if 'confusion_matrix' in self.performance_stats:
            cm = self.performance_stats['confusion_matrix']
            if true_class < len(cm) and predicted_class < len(cm[0]):
                # Return conditional probability P(true_class | predicted_class)
                col_sum = sum(cm[i][predicted_class] for i in range(len(cm)))
                if col_sum > 0:
                    return cm[true_class][predicted_class] / col_sum
                else:
                    return 0.001  # Small default probability to avoid division by zero
                    
        # Default: assume model is reasonably accurate
        if true_class == predicted_class:
            return 0.8  # Correct prediction
        else:
            return 0.2  # Incorrect prediction
    
    def _attributes_to_input(self, attributes: Dict) -> np.ndarray:
        """
        Convert attribute dictionary to model input format using preprocessing info
        
        Args:
            attributes: Dictionary of attribute values
            
        Returns:
            Numpy array suitable for model input
        """
        if not self.preprocessing_info:
            # Fallback to simple encoding
            feature_vector = []
            for attr_name in sorted(attributes.keys()):
                attr_value = attributes[attr_name]
                if isinstance(attr_value, str):
                    feature_vector.append(hash(attr_value) % 1000)
                else:
                    feature_vector.append(float(attr_value))
            return np.array(feature_vector)
        
        # Use proper preprocessing
        feature_names = self.preprocessing_info.get('feature_names', [])
        label_encoders = self.preprocessing_info.get('label_encoders', {})
        scaler = self.preprocessing_info.get('scaler', None)
        
        # Create feature vector in correct order
        feature_vector = []
        
        for feature_name in feature_names:
            if feature_name in attributes:
                attr_value = attributes[feature_name]
                
                # Apply label encoding if available
                if feature_name in label_encoders:
                    try:
                        encoded_value = label_encoders[feature_name].transform([str(attr_value)])[0]
                        feature_vector.append(encoded_value)
                    except ValueError:
                        # Handle unseen categories - use most frequent class
                        classes = label_encoders[feature_name].classes_
                        feature_vector.append(0)  # Default to first class
                else:
                    # Numerical feature
                    feature_vector.append(float(attr_value))
            else:
                # Missing feature - use default value (0)
                feature_vector.append(0.0)
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Apply scaling if available
        if scaler is not None:
            X = scaler.transform(X)
            
        return X.flatten()


class AttackEvaluator:
    """
    Evaluate the success of model inversion attacks
    """
    
    def __init__(self, attack: ModelInversionAttack):
        self.attack = attack
        
    def evaluate_attack_success(self, test_data: pd.DataFrame, target_attribute: str, 
                              known_attributes: List[str]) -> Dict:
        """
        Evaluate attack success rate on test data
        
        Args:
            test_data: DataFrame with test samples
            target_attribute: Attribute to infer
            known_attributes: List of attributes assumed to be known
            
        Returns:
            Dictionary with evaluation metrics
            e.g.:
            {
              'total_samples': 350,
              'correct_predictions': 234,
              'accuracy': 0.669,  # 66.9% attack success rate
              'baseline_accuracy': 0.596,  # Most frequent class (random guessing)
              'detailed_results': [
                  {
                      'sample_idx': 0,
                      'true_value': 'yes',
                      'predicted_value': 'yes',
                      'confidence': 0.78,
                      'correct': True,
                      'all_scores': {'yes': 0.78, 'no': 0.22}
                  },
                  # ... results for all samples
              ]
          }
        """
        results = {
            'total_samples': len(test_data),
            'correct_predictions': 0,
            'accuracy': 0.0,
            'baseline_accuracy': 0.0,
            'detailed_results': []
        }
        
        # Calculate baseline accuracy (most frequent class)
        target_values = test_data[target_attribute].values
        baseline_prediction = pd.Series(target_values).mode()[0]
        results['baseline_accuracy'] = (target_values == baseline_prediction).mean()
        
        # Test attack on each sample
        for idx, row in test_data.iterrows():
            # Extract known attributes
            known_attrs = {attr: row[attr] for attr in known_attributes}
            # what we want to infer
            true_target = row[target_attribute]
            
            # Perform attack
            predicted_target, confidence, all_scores = self.attack.infer_attribute(
                known_attrs, target_attribute
            )
            
            # Record result
            is_correct = (predicted_target == true_target)
            if is_correct:
                results['correct_predictions'] += 1
                
            results['detailed_results'].append({
                'sample_idx': idx,
                'true_value': true_target,
                'predicted_value': predicted_target,
                'confidence': confidence,
                'correct': is_correct,
                'all_scores': all_scores
            })
            
        results['accuracy'] = results['correct_predictions'] / results['total_samples']
        
        return results
    
    def compare_training_vs_validation(self, train_data: pd.DataFrame, 
                                     val_data: pd.DataFrame, target_attribute: str,
                                     known_attributes: List[str]) -> Dict:
        """
        Compare attack success on training vs validation data
        Key metric for detecting overfitting-based information leakage
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset  
            target_attribute: Attribute to infer
            known_attributes: Known attributes
            
        Returns:
            Comparison results
        """
        train_results = self.evaluate_attack_success(train_data, target_attribute, known_attributes)
        val_results = self.evaluate_attack_success(val_data, target_attribute, known_attributes)
        
        comparison = {
            'training_accuracy': train_results['accuracy'],
            'validation_accuracy': val_results['accuracy'],
            'training_advantage': train_results['accuracy'] - val_results['accuracy'],
            'baseline_accuracy': train_results['baseline_accuracy'],
            'training_over_baseline': train_results['accuracy'] - train_results['baseline_accuracy'],
            'validation_over_baseline': val_results['accuracy'] - val_results['baseline_accuracy'],
            'detailed_train': train_results,
            'detailed_validation': val_results
        }
        
        return comparison


def calculate_marginal_distributions(data: pd.DataFrame, 
                                   categorical_columns: List[str] = None) -> Dict:
    """
    Calculate marginal distributions for all attributes in dataset
    
    Args:
        data: DataFrame with training data
        categorical_columns: List of categorical column names
        
    Returns:
        Dictionary mapping attribute names to their marginal distributions
    """
    marginals = {}
    
    if categorical_columns is None:
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    
    for column in data.columns:
        if column in categorical_columns:
            # Categorical: count frequencies
            value_counts = data[column].value_counts(normalize=True)
            marginals[column] = value_counts.to_dict()
        else:
            # Numerical: check if it's actually numerical
            if pd.api.types.is_numeric_dtype(data[column]):
                # Only discretize if it's truly numerical
                try:
                    discretized = pd.cut(data[column], bins=10, duplicates='drop')
                    value_counts = discretized.value_counts(normalize=True)
                    marginals[column] = {str(interval): prob for interval, prob in value_counts.items()}
                except (TypeError, ValueError):
                    # If discretization fails, treat as categorical
                    value_counts = data[column].value_counts(normalize=True)
                    marginals[column] = value_counts.to_dict()
            else:
                # Treat as categorical if not numeric
                value_counts = data[column].value_counts(normalize=True)
                marginals[column] = value_counts.to_dict()
    
    return marginals


def calculate_model_performance_stats(model, X_test: np.ndarray, 
                                    y_test: np.ndarray) -> Dict:
    """
    Calculate model performance statistics for use in attack
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Performance statistics dictionary
    """
    predictions = model.predict(X_test)
    
    # Create confusion matrix
    unique_labels = np.unique(np.concatenate([y_test, predictions]))
    n_classes = len(unique_labels)
    confusion_matrix = np.zeros((n_classes, n_classes))
    
    for true_label, pred_label in zip(y_test, predictions):
        true_idx = np.where(unique_labels == true_label)[0][0]
        pred_idx = np.where(unique_labels == pred_label)[0][0]
        confusion_matrix[true_idx][pred_idx] += 1
    
    stats = {
        'confusion_matrix': confusion_matrix.tolist(),
        'class_labels': unique_labels.tolist(),
        'accuracy': (predictions == y_test).mean()
    }
    
    return stats