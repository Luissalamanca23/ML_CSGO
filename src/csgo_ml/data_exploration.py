"""
Data Exploration Script for CS:GO Dataset
Analyzes data characteristics, types, quality issues, and identifies problematic columns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load the raw CS:GO dataset"""
    data_path = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml\data\01_raw\Anexo_ET_demo_round_traces_2022.csv"
    df = pd.read_csv(data_path, sep=';', low_memory=False, dtype=str)  # Load all as strings first
    return df

def analyze_data_types(df):
    """Analyze data types and identify type conversion issues"""
    print("=== DATA TYPES ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print("\nColumn data types:")
    print(df.dtypes)
    
    print("\nColumns that should be numeric but are object:")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric and see if it works
            try:
                pd.to_numeric(df[col], errors='raise')
                print(f"  {col}: Could be converted to numeric")
            except:
                # Check if it contains numeric-like values
                sample_values = df[col].dropna().head(10).values
                print(f"  {col}: Sample values {sample_values}")
    
    return df.dtypes

def analyze_column_quality(df):
    """Analyze each column for data quality issues"""
    print("\n=== COLUMN QUALITY ANALYSIS ===")
    
    quality_report = {}
    
    for col in df.columns:
        print(f"\n--- {col} ---")
        
        # Basic stats
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        print(f"Null values: {null_count} ({null_pct:.2f}%)")
        print(f"Unique values: {unique_count}")
        
        # Sample values
        if df[col].dtype == 'object':
            print(f"Sample values: {df[col].value_counts().head(5).to_dict()}")
        else:
            print(f"Min: {df[col].min()}, Max: {df[col].max()}")
            print(f"Sample values: {df[col].dropna().head(5).values}")
        
        # Check for suspicious patterns
        if col in ['TimeAlive', 'TravelledDistance']:
            print("*** SUSPICIOUS VALUES DETECTED ***")
            print(f"These appear to have nonsensical large numbers:")
            print(f"Values: {df[col].head(10).values}")
        
        quality_report[col] = {
            'null_count': null_count,
            'null_pct': null_pct,
            'unique_count': unique_count,
            'dtype': str(df[col].dtype)
        }
    
    return quality_report

def identify_problematic_columns(df):
    """Identify columns that may not be useful for ML"""
    print("\n=== PROBLEMATIC COLUMNS IDENTIFICATION ===")
    
    problematic = []
    
    # Check TimeAlive and TravelledDistance for nonsensical values (they're strings with dots)
    for col in ['TimeAlive', 'TravelledDistance']:
        if col in df.columns:
            print(f"\n{col} Analysis:")
            sample_values = df[col].head(10).values
            print(f"  Sample values: {sample_values}")
            
            # Check if values contain dots (European number format issue)
            has_dots = any('.' in str(val) and len(str(val).split('.')) > 2 for val in sample_values)
            if has_dots:
                problematic.append(col)
                print(f"  *** {col} has European number format with multiple dots - unusable without conversion")
    
    # Check FirstKillTime for similar issues
    if 'FirstKillTime' in df.columns:
        print(f"\nFirstKillTime Analysis:")
        sample_values = df['FirstKillTime'].head(10).values
        print(f"  Sample values: {sample_values}")
        has_dots = any('.' in str(val) and len(str(val).split('.')) > 2 for val in sample_values if str(val) != '0.0')
        if has_dots:
            problematic.append('FirstKillTime')
            print(f"  *** FirstKillTime has European number format - needs conversion")
    
    # Check for columns with too many nulls
    for col in df.columns:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        if null_pct > 50:
            problematic.append(col)
            print(f"  *** {col} has {null_pct:.1f}% null values - may be unusable")
    
    # Check for columns with single values
    for col in df.columns:
        if df[col].nunique() <= 1:
            problematic.append(col)
            print(f"  *** {col} has only {df[col].nunique()} unique value(s) - not useful for ML")
    
    # Check RoundWinner and MatchWinner for mixed types
    for col in ['RoundWinner', 'MatchWinner']:
        if col in df.columns:
            unique_vals = df[col].unique()
            print(f"\n{col} unique values: {unique_vals}")
            if len(set(type(val).__name__ for val in unique_vals)) > 1:
                print(f"  *** {col} has mixed data types - needs cleaning")
    
    return list(set(problematic))

def analyze_target_variables(df):
    """Analyze potential target variables for classification and regression"""
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    
    # Classification targets
    print("Potential Classification Targets:")
    for col in ['RoundWinner', 'MatchWinner', 'Survived', 'Team']:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].unique()}")
            print(f"  Value counts: {df[col].value_counts().to_dict()}")
    
    # Regression targets
    print("\nPotential Regression Targets:")
    numeric_cols = ['RoundKills', 'RoundAssists', 'RoundHeadshots', 'MatchKills', 
                   'MatchAssists', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue']
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Min: {df[col].min()}, Max: {df[col].max()}")
            print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
            print(f"  Null count: {df[col].isnull().sum()}")

def generate_data_summary_report(df):
    """Generate comprehensive data summary"""
    print("\n=== DATA SUMMARY REPORT ===")
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total matches: {df['MatchId'].nunique() if 'MatchId' in df.columns else 'Unknown'}")
    print(f"Maps included: {df['Map'].unique() if 'Map' in df.columns else 'Unknown'}")
    print(f"Teams: {df['Team'].value_counts().to_dict() if 'Team' in df.columns else 'Unknown'}")

def main():
    """Main exploration function"""
    print("Starting CS:GO Data Exploration...")
    
    # Load data
    df = load_data()
    
    # Run all analyses
    analyze_data_types(df)
    quality_report = analyze_column_quality(df)
    problematic_cols = identify_problematic_columns(df)
    analyze_target_variables(df)
    generate_data_summary_report(df)
    
    print(f"\n=== FINAL RECOMMENDATIONS ===")
    print(f"Problematic columns to consider removing: {problematic_cols}")
    print(f"Total usable columns: {len(df.columns) - len(problematic_cols)}")
    
    return df, quality_report, problematic_cols

if __name__ == "__main__":
    df, quality_report, problematic_cols = main()