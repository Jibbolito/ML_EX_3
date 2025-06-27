#!/usr/bin/env python3
"""
Main experiment runner for attribute inference attacks
Tests the effectiveness of model inversion attacks on different models and datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data_preparation import load_and_prepare_dataset
from model_training import train_models_for_attack
from attack_implementation import (
    ModelInversionAttack, AttackEvaluator, 
    calculate_marginal_distributions, calculate_model_performance_stats
)


class AttributeInferenceExperiment:
    """
    Run attribute inference attack experiments on German Credit dataset
    """
    
    def __init__(self, sensitive_attributes: List[str] = None):
        """
        Initialize experiment
        
        Args:
            sensitive_attributes: List of attributes to try inferring (if None, auto-detect)
        """
        self.sensitive_attributes = sensitive_attributes
        self.data = None
        self.models = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare German Credit dataset"""
        print("Loading German Credit dataset...")
        self.data = load_and_prepare_dataset()
        
        # Auto-detect sensitive attributes if not specified
        if self.sensitive_attributes is None:
            categorical_cols = self.data['categorical_columns']
            target_col = self.data['target_column']
            
            # Use categorical columns except target as potential sensitive attributes
            self.sensitive_attributes = [col for col in categorical_cols if col != target_col]
            
        print(f"Sensitive attributes to test: {self.sensitive_attributes}")
        
    def train_models(self):
        """Train models for the experiment"""
        print("Training models...")
        trainer = train_models_for_attack(
            self.data['X_train'], self.data['y_train'],
            self.data['X_val'], self.data['y_val']
        )
        self.models = trainer
        
    def run_attack_experiments(self):
        """Run attribute inference attacks for all models and sensitive attributes"""
        print("Running attribute inference attacks...")
        
        # Calculate marginal distributions from training data
        marginals = calculate_marginal_distributions(
            self.data['train_df'], 
            self.data['categorical_columns']
        )
        
        for model_name in self.models.models.keys():
            print(f"\n--- Testing {model_name} ---")
            model = self.models.get_model(model_name)
            
            # Calculate model performance stats
            perf_stats = calculate_model_performance_stats(
                model, self.data['X_val'], self.data['y_val']
            )
            
            # Create attack instance
            attack = ModelInversionAttack(model, marginals, perf_stats, self.data['preprocessing_info'])
            evaluator = AttackEvaluator(attack)
            
            # Test each sensitive attribute
            for sensitive_attr in self.sensitive_attributes:
                print(f"  Attacking {sensitive_attr}...")
                
                # Get other attributes as "known" attributes
                known_attrs = [col for col in self.data['categorical_columns'] 
                             if col != sensitive_attr and col != self.data['target_column']]
                
                if not known_attrs:
                    print(f"    Skipping {sensitive_attr} - no other attributes available")
                    continue
                
                # Run attack comparison: training vs validation
                comparison = evaluator.compare_training_vs_validation(
                    self.data['train_df'], self.data['val_df'],
                    sensitive_attr, known_attrs
                )
                
                # Store results
                key = f"{model_name}_{sensitive_attr}"
                self.results[key] = {
                    'model_name': model_name,
                    'sensitive_attribute': sensitive_attr,
                    'known_attributes': known_attrs,
                    'comparison': comparison,
                    'model_info': self.models.get_model_info(model_name)
                }
                
                # Print summary
                print(f"    Training accuracy: {comparison['training_accuracy']:.3f}")
                print(f"    Validation accuracy: {comparison['validation_accuracy']:.3f}")
                print(f"    Training advantage: {comparison['training_advantage']:.3f}")
                print(f"    Baseline accuracy: {comparison['baseline_accuracy']:.3f}")
    
    def analyze_results(self) -> Dict:
        """Analyze and summarize experimental results"""
        print("\n" + "="*80)
        print("ATTRIBUTE INFERENCE ATTACK RESULTS")
        print("="*80)
        
        analysis = {
            'summary_stats': {},
            'best_attacks': [],
            'overfitting_correlation': [],
            'per_attribute_results': {}
        }
        
        # Create summary table
        summary_data = []
        for key, result in self.results.items():
            model_name = result['model_name']
            sensitive_attr = result['sensitive_attribute']
            comp = result['comparison']
            model_gap = result['model_info']['overfitting_gap']
            
            row = {
                'Model': model_name,
                'Sensitive_Attribute': sensitive_attr,
                'Train_Acc': comp['training_accuracy'],
                'Val_Acc': comp['validation_accuracy'],
                'Training_Advantage': comp['training_advantage'],
                'Over_Baseline': comp['training_over_baseline'],
                'Model_Overfitting_Gap': model_gap
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) > 0:
            # Print summary table
            print(summary_df.to_string(index=False, float_format='%.3f'))
            
            # Find best attacks
            best_training_advantage = summary_df.loc[summary_df['Training_Advantage'].idxmax()]
            best_over_baseline = summary_df.loc[summary_df['Over_Baseline'].idxmax()]
            
            analysis['best_attacks'] = [
                {
                    'metric': 'Training Advantage',
                    'model': best_training_advantage['Model'],
                    'attribute': best_training_advantage['Sensitive_Attribute'],
                    'value': best_training_advantage['Training_Advantage']
                },
                {
                    'metric': 'Over Baseline',
                    'model': best_over_baseline['Model'],
                    'attribute': best_over_baseline['Sensitive_Attribute'],
                    'value': best_over_baseline['Over_Baseline']
                }
            ]
            
            # Analyze correlation with overfitting
            if 'Model_Overfitting_Gap' in summary_df.columns:
                correlation = summary_df['Training_Advantage'].corr(summary_df['Model_Overfitting_Gap'])
                analysis['overfitting_correlation'].append({
                    'correlation': correlation,
                    'description': 'Correlation between model overfitting and attack success'
                })
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        
        for attack in analysis['best_attacks']:
            print(f"Best {attack['metric']}: {attack['model']} on {attack['attribute']} ({attack['value']:.3f})")
        
        if analysis['overfitting_correlation']:
            corr = analysis['overfitting_correlation'][0]['correlation']
            print(f"Correlation with overfitting: {corr:.3f}")
            if corr > 0.3:
                print("  → Strong positive correlation: More overfitting → Better attacks")
            elif corr > 0.1:
                print("  → Moderate positive correlation: Some overfitting effect")
            else:
                print("  → Weak correlation: Overfitting may not be main factor")
        
        print("="*80)
        
        return analysis
    
    def analyze_most_vulnerable_attributes(self) -> Dict:
        """Analyze which attributes are most vulnerable to inference attacks"""
        print("\n" + "="*80)
        print("DETAILED VULNERABILITY ANALYSIS")
        print("="*80)
        
        vulnerability_analysis = {}
        
        # Group results by attribute
        attr_results = {}
        for key, result in self.results.items():
            attr = result['sensitive_attribute']
            if attr not in attr_results:
                attr_results[attr] = []
            attr_results[attr].append(result['comparison'])
        
        # Analyze each attribute
        print(f"{'Attribute':<20} {'Avg Train Adv':<15} {'Max Train Adv':<15} {'Risk Level':<10}")
        print("-" * 70)
        
        for attr, results in attr_results.items():
            avg_advantage = sum(r['training_advantage'] for r in results) / len(results)
            max_advantage = max(r['training_advantage'] for r in results)
            
            # Determine risk level
            if max_advantage > 0.05:
                risk_level = "HIGH"
            elif max_advantage > 0.02:
                risk_level = "MEDIUM"
            elif max_advantage > 0:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            vulnerability_analysis[attr] = {
                'avg_training_advantage': avg_advantage,
                'max_training_advantage': max_advantage,
                'risk_level': risk_level
            }
            
            print(f"{attr:<20} {avg_advantage:<15.3f} {max_advantage:<15.3f} {risk_level:<10}")
        
        print("-" * 70)
        
        # Identify most vulnerable attributes
        high_risk_attrs = [attr for attr, data in vulnerability_analysis.items() 
                          if data['risk_level'] in ['HIGH', 'MEDIUM']]
        
        if high_risk_attrs:
            print(f"\n🚨 HIGH PRIVACY RISK ATTRIBUTES: {', '.join(high_risk_attrs)}")
            print("   These attributes show measurable information leakage!")
        else:
            print("\n✅ LOW PRIVACY RISK: Most attributes show minimal leakage")
        
        # Model-specific analysis
        print(f"\n{'Model':<20} {'Best Attribute':<20} {'Max Advantage':<15}")
        print("-" * 60)
        
        model_best = {}
        for key, result in self.results.items():
            model = result['model_name']
            attr = result['sensitive_attribute']
            advantage = result['comparison']['training_advantage']
            
            if model not in model_best or advantage > model_best[model]['advantage']:
                model_best[model] = {'attribute': attr, 'advantage': advantage}
        
        for model, data in model_best.items():
            print(f"{model:<20} {data['attribute']:<20} {data['advantage']:<15.3f}")
        
        print("="*80)
        return vulnerability_analysis
    
    def plot_results(self, save_path: str = None):
        """Create visualizations of attack results"""
        if not self.results:
            print("No results to plot")
            return
            
        # Prepare data for plotting
        plot_data = []
        for key, result in self.results.items():
            comp = result['comparison']
            model_info = result['model_info']
            
            plot_data.append({
                'Model': result['model_name'],
                'Attribute': result['sensitive_attribute'],
                'Training_Accuracy': comp['training_accuracy'],
                'Validation_Accuracy': comp['validation_accuracy'],
                'Training_Advantage': comp['training_advantage'],
                'Overfitting_Gap': model_info['overfitting_gap']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) == 0:
            print("No data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Attribute Inference Attack Results - German Credit Dataset', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Training vs Validation Accuracy
        ax1 = axes[0, 0]
        models = plot_df['Model'].unique()
        x_pos = np.arange(len(models))
        
        train_accs = [plot_df[plot_df['Model'] == m]['Training_Accuracy'].mean() for m in models]
        val_accs = [plot_df[plot_df['Model'] == m]['Validation_Accuracy'].mean() for m in models]
        
        width = 0.35
        ax1.bar(x_pos - width/2, train_accs, width, label='Training', alpha=0.8)
        ax1.bar(x_pos + width/2, val_accs, width, label='Validation', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Attack Accuracy')
        ax1.set_title('Attack Accuracy: Training vs Validation')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Advantage by Model
        ax2 = axes[0, 1]
        advantages = [plot_df[plot_df['Model'] == m]['Training_Advantage'].mean() for m in models]
        bars = ax2.bar(models, advantages, alpha=0.8, color='coral')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Training Advantage')
        ax2.set_title('Training Advantage by Model Type')
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Highlight best performing model
        max_idx = np.argmax(advantages)
        bars[max_idx].set_color('red')
        bars[max_idx].set_alpha(1.0)
        
        # Plot 3: Overfitting vs Attack Success
        ax3 = axes[1, 0]
        if 'Overfitting_Gap' in plot_df.columns and plot_df['Overfitting_Gap'].notna().any():
            scatter = ax3.scatter(plot_df['Overfitting_Gap'], plot_df['Training_Advantage'], 
                                c=range(len(plot_df)), cmap='viridis', alpha=0.7)
            ax3.set_xlabel('Model Overfitting Gap')
            ax3.set_ylabel('Training Advantage')
            ax3.set_title('Overfitting vs Attack Success')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(plot_df['Overfitting_Gap'], plot_df['Training_Advantage'], 1)
            p = np.poly1d(z)
            ax3.plot(plot_df['Overfitting_Gap'], p(plot_df['Overfitting_Gap']), "r--", alpha=0.8)
        else:
            ax3.text(0.5, 0.5, 'No overfitting data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Overfitting vs Attack Success')
        
        # Plot 4: Attack Success by Attribute
        ax4 = axes[1, 1]
        if len(plot_df['Attribute'].unique()) > 1:
            attr_advantages = plot_df.groupby('Attribute')['Training_Advantage'].mean().sort_values(ascending=False)
            bars = ax4.bar(range(len(attr_advantages)), attr_advantages.values, alpha=0.8, color='lightblue')
            ax4.set_xlabel('Sensitive Attribute')
            ax4.set_ylabel('Training Advantage')
            ax4.set_title('Attack Success by Sensitive Attribute')
            ax4.set_xticks(range(len(attr_advantages)))
            ax4.set_xticklabels(attr_advantages.index, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Highlight most vulnerable attribute
            bars[0].set_color('red')
            bars[0].set_alpha(1.0)
        else:
            ax4.text(0.5, 0.5, f'Single attribute tested:\n{plot_df["Attribute"].iloc[0]}', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Attack Success by Sensitive Attribute')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Don't show plot interactively to avoid blocking
        plt.close('all')
    
    def run_full_experiment(self, save_plots: bool = True) -> Dict:
        """Run complete experiment pipeline"""
        print("="*80)
        print("STARTING ATTRIBUTE INFERENCE ATTACK EXPERIMENT")
        print("Dataset: German Credit")
        print("="*80)
        
        # Run experiment steps
        self.load_data()
        self.train_models()
        self.run_attack_experiments()
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Enhanced vulnerability analysis
        vulnerability_analysis = self.analyze_most_vulnerable_attributes()
        analysis['vulnerability_analysis'] = vulnerability_analysis
        
        # Create plots
        if save_plots:
            plot_path = "attack_results_german_credit.png"
            self.plot_results(plot_path)
        
        return analysis


if __name__ == "__main__":
    # Run attribute inference attack experiment on German Credit dataset
    print("Testing attribute inference attack implementation...")
    
    experiment = AttributeInferenceExperiment()
    results = experiment.run_full_experiment()
    
    print("\nExperiment completed successfully!")
    print("German Credit dataset attribute inference attack analysis complete.")