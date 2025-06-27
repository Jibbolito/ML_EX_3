#!/usr/bin/env python3
"""
Test implementation of attribute inference attack components
Validates that all parts of the Fredrikson algorithm work correctly before running experiments
"""

import sys
import traceback
import numpy as np
import pandas as pd

def test_data_loading():
    """Test German Credit dataset loading and preprocessing"""
    print("1. Testing data loading...")
    try:
        from data_preparation import load_and_prepare_dataset
        data = load_and_prepare_dataset()
        
        # Validate data structure
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
                        'train_df', 'val_df', 'test_df', 'preprocessing_info',
                        'categorical_columns', 'target_column']
        
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        # Validate shapes and content
        assert data['X_train'].shape[0] > 0, "No training data"
        assert data['X_val'].shape[0] > 0, "No validation data"
        assert len(data['categorical_columns']) > 0, "No categorical columns"
        assert data['target_column'] == 'class', "Wrong target column"
        
        print(f"   ✓ German Credit dataset loaded successfully")
        print(f"   ✓ Training samples: {data['X_train'].shape[0]}")
        print(f"   ✓ Validation samples: {data['X_val'].shape[0]}")
        print(f"   ✓ Features: {data['X_train'].shape[1]}")
        print(f"   ✓ Sensitive attributes: {len(data['categorical_columns'])}")
        return data
        
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        traceback.print_exc()
        return None

def test_model_training(data):
    """Test training of overfitting-prone models"""
    print("\n2. Testing model training...")
    try:
        from model_training import train_models_for_attack
        
        trainer = train_models_for_attack(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        # Validate all required models exist
        expected_models = ['decision_tree', 'neural_network', 'random_forest', 'logistic_regression']
        for model_name in expected_models:
            model = trainer.get_model(model_name)
            assert hasattr(model, 'predict'), f"Model {model_name} missing predict method"
            assert hasattr(model, 'predict_proba'), f"Model {model_name} missing predict_proba method"
        
        # Test model predictions work
        test_sample = data['X_val'][:1]
        for model_name in expected_models:
            model = trainer.get_model(model_name)
            pred = model.predict(test_sample)
            proba = model.predict_proba(test_sample)
            assert len(pred) == 1, f"{model_name} prediction shape wrong"
            assert proba.shape == (1, 2), f"{model_name} probability shape wrong"
        
        # Check overfitting occurred
        overfitting_models = ['decision_tree', 'neural_network', 'random_forest']
        for model_name in overfitting_models:
            model_info = trainer.get_model_info(model_name)
            gap = model_info['overfitting_gap']
            assert gap > 0.1, f"{model_name} not overfitting enough (gap: {gap})"
        
        print(f"   ✓ All models trained successfully")
        print(f"   ✓ Models available: {list(trainer.models.keys())}")
        print(f"   ✓ Overfitting achieved (max gap: {max(trainer.get_model_info(m)['overfitting_gap'] for m in overfitting_models):.3f})")
        return trainer
        
    except Exception as e:
        print(f"   ✗ Model training failed: {e}")
        traceback.print_exc()
        return None

def test_attack_components(data, trainer):
    """Test core attack implementation components"""
    print("\n3. Testing attack components...")
    try:
        from attack_implementation import (
            ModelInversionAttack, calculate_marginal_distributions, 
            calculate_model_performance_stats
        )
        
        # Test marginal distributions
        marginals = calculate_marginal_distributions(
            data['train_df'], data['categorical_columns']
        )
        assert len(marginals) > 0, "No marginal distributions calculated"
        
        # Validate marginals for categorical columns
        for col in data['categorical_columns'][:3]:  # Test first 3
            if col in data['train_df'].columns:
                assert col in marginals, f"Missing marginals for {col}"
                prob_sum = sum(marginals[col].values())
                assert 0.95 <= prob_sum <= 1.05, f"Invalid probabilities for {col}"
        
        # Test performance statistics
        model = trainer.get_model('decision_tree')
        perf_stats = calculate_model_performance_stats(
            model, data['X_val'], data['y_val']
        )
        assert 'confusion_matrix' in perf_stats, "Missing confusion matrix"
        assert 'accuracy' in perf_stats, "Missing accuracy"
        
        # Test attack instance creation
        attack = ModelInversionAttack(
            model, marginals, perf_stats, data['preprocessing_info']
        )
        
        # Test feature conversion
        if len(data['categorical_columns']) >= 2:
            test_attrs = {}
            for col in data['categorical_columns'][:2]:
                if col in data['train_df'].columns:
                    valid_values = data['train_df'][col].dropna().unique()
                    if len(valid_values) > 0:
                        test_attrs[col] = valid_values[0]
            
            if test_attrs:
                X_converted = attack._attributes_to_input(test_attrs)
                expected_features = data['X_train'].shape[1]
                assert len(X_converted) == expected_features, f"Feature mismatch: {len(X_converted)} vs {expected_features}"
                
                # Test model can handle converted features
                pred_proba = model.predict_proba(X_converted.reshape(1, -1))
                assert pred_proba.shape == (1, 2), "Converted features prediction failed"
        
        print(f"   ✓ Marginal distributions calculated ({len(marginals)} attributes)")
        print(f"   ✓ Model performance statistics computed")
        print(f"   ✓ Attack instance created successfully")
        print(f"   ✓ Feature conversion working")
        return attack
        
    except Exception as e:
        print(f"   ✗ Attack components failed: {e}")
        traceback.print_exc()
        return None

def test_attack_execution(data, attack):
    """Test actual attribute inference attack execution"""
    print("\n4. Testing attack execution...")
    try:
        from attack_implementation import AttackEvaluator
        
        evaluator = AttackEvaluator(attack)
        
        # Find suitable attributes for testing
        available_attrs = [col for col in data['categorical_columns'] 
                          if col in data['train_df'].columns and col != data['target_column']]
        
        if len(available_attrs) < 2:
            print("   ⚠ Not enough attributes for attack testing")
            return True
            
        # Test attack on small subset
        target_attr = available_attrs[0]
        known_attrs = [available_attrs[1]] if len(available_attrs) > 1 else []
        
        if not known_attrs:
            print("   ⚠ No known attributes available for attack")
            return True
        
        # Test on small sample
        small_sample = data['train_df'].head(5)
        results = evaluator.evaluate_attack_success(small_sample, target_attr, known_attrs)
        
        # Validate results structure
        required_metrics = ['total_samples', 'correct_predictions', 'accuracy', 'baseline_accuracy']
        for metric in required_metrics:
            assert metric in results, f"Missing metric: {metric}"
        
        assert results['total_samples'] == 5, "Wrong sample count"
        assert 0 <= results['accuracy'] <= 1, "Invalid accuracy range"
        assert 0 <= results['baseline_accuracy'] <= 1, "Invalid baseline accuracy"
        
        # Test training vs validation comparison
        comparison = evaluator.compare_training_vs_validation(
            data['train_df'].head(10), data['val_df'].head(10),
            target_attr, known_attrs
        )
        
        required_comparison_keys = ['training_accuracy', 'validation_accuracy', 'training_advantage']
        for key in required_comparison_keys:
            assert key in comparison, f"Missing comparison key: {key}"
        
        print(f"   ✓ Attack execution successful")
        print(f"   ✓ Testing attribute: {target_attr}")
        print(f"   ✓ Attack accuracy: {results['accuracy']:.3f}")
        print(f"   ✓ Baseline accuracy: {results['baseline_accuracy']:.3f}")
        print(f"   ✓ Training vs validation comparison working")
        return True
        
    except Exception as e:
        print(f"   ✗ Attack execution failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Execute all validation tests for the attribute inference implementation"""
    print("="*80)
    print("ATTRIBUTE INFERENCE ATTACK - IMPLEMENTATION VALIDATION")
    print("Testing Fredrikson et al. methodology on German Credit dataset")
    print("="*80)
    
    # Test sequence
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Data Loading
    data = test_data_loading()
    if data is None:
        print("\n❌ CRITICAL FAILURE: Cannot load German Credit dataset")
        return False
    tests_passed += 1
    
    # Test 2: Model Training  
    trainer = test_model_training(data)
    if trainer is None:
        print("\n❌ CRITICAL FAILURE: Cannot train overfitting models")
        return False
    tests_passed += 1
    
    # Test 3: Attack Components
    attack = test_attack_components(data, trainer)
    if attack is None:
        print("\n❌ CRITICAL FAILURE: Attack components not working")
        return False
    tests_passed += 1
    
    # Test 4: Attack Execution
    attack_success = test_attack_execution(data, attack)
    if not attack_success:
        print("\n❌ CRITICAL FAILURE: Attack execution failed")
        return False
    tests_passed += 1
    
    # Final validation
    print("\n" + "="*80)
    print(f"✅ ALL TESTS PASSED ({tests_passed}/{total_tests})")
    print("✅ Implementation validated - ready for full experiment")
    print("✅ Run: python experiments.py")
    print("="*80)
    return True

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)