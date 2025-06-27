#!/usr/bin/env python3
"""
Data preparation for attribute inference attack experiments
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class DatasetLoader:
    """
    Load and prepare German Credit dataset for attribute inference attacks
    """
    
    @staticmethod # Stateless Operation , stnadalone utility function, better performance
    def load_german_credit_dataset() -> Tuple[pd.DataFrame, str, List[str]]:
        """
        Load German Credit dataset
        Good for testing inference of personal attributes
        
        Returns:
            Tuple of (dataframe, target_column, categorical_columns)
        """
        try:
            # Load German Credit dataset
            credit = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
            df = credit.frame
            
            # Clean column names
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            # Convert numerical columns to proper numeric types
            numeric_cols = ['duration', 'credit_amount', 'installment_commitment', 'residence_since',
                           'age', 'existing_credits', 'num_dependents']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Define categorical columns (exclude numeric and target columns)
            categorical_cols = [col for col in df.columns 
                              if col not in numeric_cols and col != 'class']
            
            target_col = 'class'
            
            print(f"German Credit dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            print(f"Categorical columns: {categorical_cols}")
            
            return df, target_col, categorical_cols
            
        except Exception as e:
            print(f"Error loading German Credit dataset: {e}")
            return None, None, None
    


class DataPreprocessor:
    """
    Preprocess datasets for model training and attack evaluation
    """
    
    def __init__(self, encode_categoricals: bool = True):
        self.encode_categoricals = encode_categoricals
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def prepare_for_modeling(self, df: pd.DataFrame, target_col: str, 
                           categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Prepare dataset for model training
        
        Args:
            df: Raw dataframe
            target_col: Target column name
            categorical_cols: List of categorical columns
            
        Returns:
            Tuple of (X, y, preprocessing_info)
        """
        df_processed = df.copy()
        preprocessing_info = {
            'target_column': target_col,
            'categorical_columns': categorical_cols,
            'feature_names': [],
            'label_encoders': {}
        }
        
        # Encode categorical variables
        if self.encode_categoricals:
            for col in categorical_cols:
                if col in df_processed.columns and col != target_col:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    preprocessing_info['label_encoders'][col] = le
        
        # Encode target variable
        if target_col in df_processed.columns:
            target_le = LabelEncoder()
            y = target_le.fit_transform(df_processed[target_col].astype(str))
            preprocessing_info['target_encoder'] = target_le
            self.label_encoders[target_col] = target_le
        else:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Prepare features
        X_cols = [col for col in df_processed.columns if col != target_col]
        X = df_processed[X_cols].values
        preprocessing_info['feature_names'] = X_cols
        
        # Scale numerical features
        numerical_cols = [col for col in X_cols if col not in categorical_cols]
        if numerical_cols:
            X = self.scaler.fit_transform(X)
            preprocessing_info['scaler'] = self.scaler
        
        return X, y, preprocessing_info
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.3, val_size: float = 0.5) -> Tuple:
        """
        Split data into train/validation/test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        print(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare_dataset() -> Tuple:
    """
    Load and prepare German Credit dataset for attribute inference experiments
    
    Returns:
        Tuple containing processed data and metadata
    """
    loader = DatasetLoader()
    preprocessor = DataPreprocessor()
    
    # Load German Credit dataset
    df, target_col, categorical_cols = loader.load_german_credit_dataset()
    
    if df is None:
        raise ValueError("Failed to load German Credit dataset")
    
    # Prepare for modeling
    X, y, preprocessing_info = preprocessor.prepare_for_modeling(df, target_col, categorical_cols)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Create proper train/val/test split for DataFrames
    # Reset index to ensure proper alignment
    df_reset = df.reset_index(drop=True)
    
    # First split: temp vs test
    df_temp, test_df = train_test_split(df_reset, test_size=0.3, random_state=42, 
                                       stratify=df_reset[target_col] if target_col in df_reset.columns else None)
    
    # Second split: train vs val  
    train_df, val_df = train_test_split(df_temp, test_size=0.5, random_state=42,
                                       stratify=df_temp[target_col] if target_col in df_temp.columns else None)
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'preprocessing_info': preprocessing_info,
        'categorical_columns': categorical_cols,
        'target_column': target_col,
        'raw_df': df
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing German Credit dataset loading...")
    
    try:
        data = load_and_prepare_dataset()
        print("Successfully loaded German Credit dataset")
        print(f"Features shape: {data['X_train'].shape}")
        print(f"Target distribution: {np.bincount(data['y_train'])}")
        print(f"Sensitive attributes: {data['categorical_columns']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")