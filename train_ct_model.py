
#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import random
from scipy import stats
import argparse

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def concordance_index_scorer(estimator, X, y):
    """
    Computes the concordance index for a survival model's predictions.

    Parameters:
        estimator: Trained survival model with a .predict() method.
        X: Feature matrix for prediction.
        y: Structured array with fields 'event' and 'time'.

    Returns:
        Concordance index (float)
    """
    event, time = y["event"], y["time"]
    prediction = estimator.predict(X)
    return concordance_index_censored(event, time, prediction)[0]

def confidence_interval(cindex_scores, confidence=0.95):
    """
    Computes the mean and confidence interval for a list of C-index scores.

    Parameters:
        cindex_scores: List or array of C-index values.
        confidence: Confidence level (default is 95%).

    Returns:
        Tuple: (mean, lower_bound, upper_bound)
    """
    mean = np.mean(cindex_scores)
    sem = stats.sem(cindex_scores)
    ci = stats.t.interval(confidence, df=len(cindex_scores)-1, loc=mean, scale=sem)
    print(f"Mean C-index: {mean:.2f}, {int(confidence*100)}% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
    

def clinical_data(train_cohort, test_cohort, clinical_f=None):
    """
    Encodes categorical clinical variables for training and testing datasets.

    Parameters:
    - train_cohort: DataFrame for training data
    - test_cohort: DataFrame for test data
    - clinical_f: Not used, kept for compatibility

    Returns:
    - clinical_train_df: DataFrame with encoded clinical features for training
    - clinical_test_df: DataFrame with encoded clinical features for testing
    """
    
    # Define mappings
    sex_map = {'Female': 1, 'Male': 0}
    smoking_map = {
        'Former smoker': 0,
        'Non smoker': 1,
        'Smoker': 2,
        'Passive smoker': 3
    }
    histo_map = {
        'Carcinoid tumor': 0,
        'Small cell carcinoma': 1,
        'Large cell neuroendocrine carcinoma': 2
    }

    # Apply mappings
    def encode_feature(df, column, mapping):
        return df[column].map(mapping).values.reshape(-1, 1)

    train_sex = encode_feature(train_cohort, 'sex', sex_map)
    test_sex = encode_feature(test_cohort, 'sex', sex_map)

    train_smoking = encode_feature(train_cohort, 'smoking_habit', smoking_map)
    test_smoking = encode_feature(test_cohort, 'smoking_habit', smoking_map)

    train_histo = encode_feature(train_cohort, 'Histological subtype 1_x', histo_map)
    test_histo = encode_feature(test_cohort, 'Histological subtype 1_x', histo_map)

    # Concatenate features
    clinical_train = np.hstack([train_sex, train_smoking, train_histo])
    clinical_test = np.hstack([test_sex, test_smoking, test_histo])

    # Create DataFrames
    columns = ['sex', 'smoking_habit', 'Histological subtype 1_x']
    clinical_train_df = pd.DataFrame(clinical_train, columns=columns)
    clinical_test_df = pd.DataFrame(clinical_test, columns=columns)

    #print('clinical_train_df shape:', clinical_train_df.shape)
    return clinical_train_df, clinical_test_df



def fit_and_score_features(X, y):
    """
    Scores each feature independently using a univariate CoxnetSurvivalAnalysis.

    Parameters:
    - X: 2D array-like, shape (n_samples, n_features)
    - y: structured array with fields ('event', 'time')

    Returns:
    - scores: array of concordance scores for each feature
    """
    n_features = X.shape[1]
    scores = np.empty(n_features)
    model = CoxnetSurvivalAnalysis(l1_ratio=0.1, alpha_min_ratio=0.001)

    for j in range(n_features):
        Xj = X[:, j:j+1]
        model.fit(Xj, y)
        scores[j] = model.score(Xj, y)

    return scores

def remove_highly_correlated_features(X, y, target_column, threshold=0.9):
    """
    Remove features from X that are highly correlated with each other,
    keeping the one more correlated with the target.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series or pd.DataFrame, target variable
    - target_column: str, name of the target column
    - threshold: float, correlation threshold for dropping features

    Returns:
    - X_new: pd.DataFrame, with selected features
    """
    # Combine features and target into one DataFrame
    df = pd.concat([X, y], axis=1)

    # Compute correlation matrix
    correlation_matrix = df.corr()

    # Extract upper triangle of the correlation matrix
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Identify pairs with high correlation
    high_corr_pairs = [
        (col1, col2) for col1 in upper_triangle.columns
        for col2 in upper_triangle.index
        if upper_triangle.at[col2, col1] > threshold
    ]

    # Decide which columns to drop
    columns_to_drop = set()
    for col1, col2 in high_corr_pairs:
        if abs(correlation_matrix.at[target_column, col1]) >= abs(correlation_matrix.at[target_column, col2]):
            columns_to_drop.add(col2)
        else:
            columns_to_drop.add(col1)

    return columns_to_drop


def load_data(input_file, target_column, vital_status_column, clinical_features):
   
    df = pd.read_csv(input_file)
    df = df[df[target_column].notna() & (df[target_column] != 0)].reset_index(drop=True)

    # Prepare features and labels
    X = df.iloc[:, :-9]
    y = pd.DataFrame(df[target_column])
    vital = pd.DataFrame(df['Vital status'].to_list(), columns=['status'])
    vital['status'] = vital['status'].replace({'Alive': 1, 'Deceased': 0})
    clinical = df[['sex', 'smoking_habit', 'Histological subtype 1_x']]

    return X, y,vital, clinical

def main(args):
    X, y,vital, clinical = load_data(
        args.input_file,
        target_column=args.target_column,
        vital_status_column=args.vital_status_column,
        clinical_features=args.clinical_features.split(',')
    )

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = random.sample(range(1, 1000), args.n_seeds)

    print(f"Using seeds: {seeds}")
    result_all = [{'seed': 0, 'cv_cindex_rad': 0, 'test_cindex_rad': 0, 'hr_rad': 0,
                'best_features': 0, 'cv_cindex_clin': 0, 'test_cindex_clin': 0, 'hr_clin': 0}]
    for seed in seeds:
        # Split data
        X_train, X_test, y_train, y_test, clin_train, clin_test, vital_train, vital_test = train_test_split(
            X, y, clinical, vital, test_size=0.30, random_state=seed, stratify=vital)

        # Convert to DataFrames with proper columns
        X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
        X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
        clin_train = pd.DataFrame(clin_train, columns=clinical.columns).reset_index(drop=True)
        clin_test = pd.DataFrame(clin_test, columns=clinical.columns).reset_index(drop=True)
        y_train = pd.DataFrame(y_train, columns=y.columns).reset_index(drop=True)
        y_test = pd.DataFrame(y_test, columns=y.columns).reset_index(drop=True)
        vital_train = pd.DataFrame(vital_train, columns=vital.columns).reset_index(drop=True)
        vital_test = pd.DataFrame(vital_test, columns=vital.columns).reset_index(drop=True)

        # Prepare survival labels
        data_y_train = Surv.from_arrays(event=vital_train.status.tolist(), time=y_train[args.target_column].tolist())
        data_y_test = Surv.from_arrays(event=vital_test.status.tolist(), time=y_test[args.target_column].tolist())

        # Feature selection
        columns_to_drop = remove_highly_correlated_features(X_train, y_train, target_column=args.target_column)
        X_train_filtered = X_train.drop(columns=columns_to_drop, errors='ignore')
        X_test_filtered = X_test.drop(columns=columns_to_drop, errors='ignore')

        # Radiomics model pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()), 
            ("select", SelectKBest(fit_and_score_features, k=10)),
            ("model", CoxnetSurvivalAnalysis()),
        ])

        param_grid = {
            "select__k": np.arange(1, 15),
            "model__l1_ratio": [ 0.3, 0.4, 0.5,0.6],
        }

        cv = KFold(n_splits=3, random_state=1, shuffle=True)
        rad_model = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv, scoring=concordance_index_scorer)
        rad_model.fit(X_train_filtered, data_y_train)

        # Evaluate radiomics model
        best_rad = rad_model.best_estimator_
        rad_cv = rad_model.best_score_
        predicted_risk = best_rad.predict(X_test_filtered)
        rad_test_score = concordance_index_censored(
            vital_test.status.astype(bool).to_numpy(), y_test[args.target_column].to_numpy(), predicted_risk)
        hr_rad = np.mean(np.exp(best_rad.named_steps["model"].coef_))

        # Feature importance
        best_k = rad_model.best_params_['select__k']
        ranking_feat = fit_and_score_features(X_train_filtered.values, data_y_train)
        top_features = list(pd.Series(ranking_feat, index=X_train_filtered.columns)
                            .sort_values(ascending=False)[:best_k].keys())

        x_train_fs = X_train_filtered[top_features]
        x_test_fs = X_test_filtered[top_features]

        # Clinical model
        clinical_f = 'all_exceptage'
        clinical_train, clinical_test = clinical_data(clin_train, clin_test, clinical_f)

        comb_train = pd.concat([x_train_fs, clinical_train], axis=1)
        comb_test = pd.concat([x_test_fs, clinical_test], axis=1)

        pipe_clin = Pipeline([
            ("scaler", StandardScaler()),
            ("model", CoxnetSurvivalAnalysis()),
        ])

        param_grid_clin = {
            "model__l1_ratio": [0.0003, 0.0004,0.0005, 0.0007, 0.0009],
        }
        comb_model = GridSearchCV(pipe_clin, param_grid_clin, return_train_score=True, cv=cv, scoring=concordance_index_scorer)
        comb_model.fit(comb_train, data_y_train)

        # Evaluate combined model
        best_comb = comb_model.best_estimator_
        predicted_risk_comb = best_comb.predict(comb_test)
        comb_test_score = concordance_index_censored(
            vital_test.status.astype(bool).to_numpy(), y_test[args.target_column].to_numpy(), predicted_risk_comb)
        hr_cl = np.mean(np.exp(best_comb.named_steps["model"].coef_))
        comb_cv = comb_model.best_score_

        # Save results
        result_all.append({
            'seed': seed,
            'cv_cindex_rad': rad_cv,
            'test_cindex_rad': rad_test_score[0],
            'hr_rad': hr_rad,
            'best_features': len(top_features),
            'cv_cindex_clin': comb_cv,
            'test_cindex_clin': comb_test_score[0],
            'hr_clin': hr_cl
        })


    df_results = pd.DataFrame(result_all[1:])
    output_filename = f"result_ct_{args.target_column.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    df_results.to_csv(output_filename, index=False)
    print('End')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run survival analysis with Coxnet and compute C-index.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--target_column", type=str, default="O.S. (2022)", choices=["O.S. (2022)", "PFS (2022)"],
 help="Target survival time column")
    parser.add_argument("--vital_status_column", type=str, default="Vital status", help="Vital status column")
    parser.add_argument("--clinical_features", type=str, default="age,sex,smoking_habit,Histological subtype 1_x", help="Comma-separated clinical features to exclude from X")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--seeds", type=str, help="Comma-separated list of seeds (optional)")

    args = parser.parse_args()
    main(args)


