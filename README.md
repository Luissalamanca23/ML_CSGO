# CS:GO Machine Learning Analysis

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Project Overview

This project demonstrates comprehensive machine learning methodologies applied to Counter-Strike: Global Offensive (CS:GO) gameplay data. The analysis follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to develop predictive models for player performance and match outcomes.

**Business Context**: Analysis of 79,157 gameplay records from 333 professional CS:GO matches to derive actionable insights for competitive gaming strategies and player performance optimization.

**Technical Stack**: Python-based data science pipeline utilizing Kedro framework, scikit-learn, XGBoost, and comprehensive statistical analysis libraries.

## Data Science Methodology

### CRISP-DM Implementation

**1. Business Understanding**
- Regression Problem: Predicting player performance metrics (RoundKills, Equipment ROI)
- Classification Problem: Predicting player survival probability and match outcomes
- Domain expertise integration: CS:GO game mechanics and tactical considerations

**2. Data Understanding**
- Raw dataset: 79,157 observations across 30 variables from professional match replays
- Temporal scope: 333 matches across 4 official competitive maps
- Data quality assessment: Identification of corrupted European number formats and inconsistent data types

**3. Data Preparation**
- Data cleaning: Removal of unusable temporal variables (TimeAlive, TravelledDistance) due to format corruption
- Feature engineering: Creation of 78+ derived features including tactical, economic, and performance indicators
- Correlation analysis: Systematic elimination of multicollinear features to prevent model degradation

**4. Modeling**
- Regression models: Random Forest, Gradient Boosting, XGBoost, Ridge, SVR with GridSearchCV optimization
- Classification models: Random Forest, Gradient Boosting, XGBoost, Logistic Regression, SVC with ROC analysis
- Cross-validation: 5-fold stratified validation with comprehensive overfitting detection

### Data Architecture

**Data Layer Structure** (following Kedro conventions):
- `01_raw/`: Original match replay data (Anexo_ET_demo_round_traces_2022.csv)
- `02_intermediate/`: Cleaned datasets with quality issues resolved
- `03_primary/`: Validated and standardized data ready for analysis
- `04_feature/`: Enhanced datasets with engineered features (120+ variables)
- `05_model_input/`: Preprocessed data optimized for machine learning algorithms
- `06_models/`: Trained model artifacts with hyperparameter configurations
- `07_model_output/`: Predictions, performance metrics, and evaluation results
- `08_reporting/`: Analytical visualizations and business insights

### Technical Implementation

**Feature Engineering Strategies:**
- Performance metrics: Kill-to-equipment efficiency ratios, tactical effectiveness indicators
- Temporal features: Round progression phases, economic cycle patterns
- Interactive features: Equipment-performance synergies, team-map advantage combinations
- Statistical aggregations: Rolling performance windows, cumulative statistics

**Model Selection Criteria:**
- Regression target: R² > 0.8 with residual analysis for homoscedasticity
- Classification target: ROC AUC > 0.8 with precision-recall optimization for imbalanced classes
- Overfitting prevention: Train-test gap monitoring and regularization parameter tuning

## Project Structure and Dependencies

## Installation and Setup

### Prerequisites
- Python 3.8+
- Virtual environment management (recommended)

### Dependency Installation

Install project dependencies:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- kedro==0.19.14 (Pipeline orchestration)
- scikit-learn (Machine learning algorithms)
- xgboost (Gradient boosting framework)
- pandas, numpy (Data manipulation)
- matplotlib, seaborn, plotly (Visualization)
- pytest (Testing framework)

## Execution Workflows

### Pipeline Execution

Execute the complete CRISP-DM workflow:

```bash
kedro run
```

**Individual Pipeline Components:**
```bash
# Data processing and preparation
kedro run --pipeline data_processing

# Machine learning modeling
kedro run --pipeline data_science

# Reporting and visualization
kedro run --pipeline reporting
```

### Jupyter Analysis Environment

Access interactive analysis environment with full Kedro context:

```bash
# Jupyter Notebook
kedro jupyter notebook

# JupyterLab interface
kedro jupyter lab

# IPython session with catalog access
kedro ipython
```

### Key Notebooks

- `data_preparation_eda.ipynb`: Comprehensive exploratory data analysis and preprocessing
- `feature_engineering.ipynb`: Advanced feature creation and correlation analysis
- `regression_modeling.ipynb`: Performance prediction models with GridSearchCV optimization
- `classification_modeling.ipynb`: Survival and outcome prediction with ROC analysis

## Quality Assurance

### Testing Framework

Execute comprehensive test suite:

```bash
pytest
```

**Test Coverage:**
- Data pipeline validation
- Model performance benchmarks
- Feature engineering consistency checks

Coverage configuration available in `.coveragerc` file.

## Model Performance Benchmarks

### Regression Models (Target: RoundKills)
- **Objective**: R² > 0.8 with residual normality verification
- **Features**: 15 engineered variables focusing on equipment efficiency and tactical indicators
- **Validation**: 5-fold cross-validation with overfitting detection (train-test gap < 0.1)

### Classification Models (Target: Player Survival)
- **Objective**: ROC AUC > 0.8 with balanced precision-recall optimization
- **Features**: 15 strategic variables excluding match-level leakage predictors
- **Evaluation**: Comprehensive ROC analysis, confusion matrices, and threshold optimization

## Project Deliverables

### Documentation
- **Business Understanding**: Regression and classification problem definitions with domain-specific requirements
- **Data Understanding**: Comprehensive data quality assessment with corruption identification and resolution strategies
- **Technical Documentation**: Feature engineering methodology and model selection rationale

### Analysis Notebooks
- **EDA Pipeline**: Statistical analysis, correlation matrices, and data quality visualization
- **Feature Engineering**: Advanced feature creation with domain knowledge integration
- **Model Development**: Hyperparameter optimization with GridSearchCV and performance benchmarking

### Model Artifacts
- **Trained Models**: Optimized algorithms with validated hyperparameters
- **Performance Metrics**: Cross-validation results, learning curves, and overfitting analysis
- **Prediction Outputs**: Model predictions with confidence intervals and feature importance rankings

## Technical Standards

**Code Quality**: PEP 8 compliance with comprehensive docstrings and type annotations  
**Reproducibility**: Seed management and environment specification for consistent results  
**Version Control**: Git workflow with data lineage tracking and model versioning  
**Documentation**: Technical specifications following data science best practices

## References

- [Kedro Documentation](https://docs.kedro.org) - Pipeline orchestration framework
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/) - Data mining process standard
- [CS:GO Game Mechanics](https://blog.counter-strike.net/) - Domain knowledge source
