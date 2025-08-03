import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, 
                             matthews_corrcoef, brier_score_loss)

# ---- Define Helper Functions ----

def run_grid_search(args, X_train, y_train, MODELS):
    """Run grid search and return the best parameters for each model."""
    if args["run_grid_search"]:
        print("Running grid search for hyperparameter tuning ..", end='\r')
        # Define the parameter grid for grid search
        param_grid = {
            'Logistic Regression': { 'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            'Multinomial NB': { 'clf__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
            'Random Forest': { 'clf__n_estimators': [10, 30]},
            'Model SDI': {
                'clf__alpha': [0.01, 0.1, 0.5, 1.0], 
                'clf__beta': [0.05, 0.25, 0.5, 0.75], 
                'clf__eps': [1e-5, 1e-3], 
                'clf__initialization': [0], 
                'clf__max_iter': [100], 
                'clf__unsupervised_perc': [1.0]
            },
        }

        model2parameters = {}
        for model_name, param_grid in param_grid.items():
            grid_search = GridSearchCV(MODELS[model_name], param_grid, cv=3)
            grid_search.fit(X_train, y_train)
            model2parameters[model_name] = grid_search.best_params_
        print("Running grid search for hyperparameter tuning ✅")
        return model2parameters
    else:
        print(" Skipping parameters grid search")
        return None

def calculate_metrics(MODEL2y_pred, y_true):
    """Calculate metrics and return a DataFrame with results."""
    df_data = {"metric_name": [], "metric_value": [], "model_name": []}

    for model_name, y_score in MODEL2y_pred.items():
        metrics_values = [
            accuracy_score(y_true, y_score>.5),
            f1_score(y_true, y_score>.5),
            roc_auc_score(y_true, y_score),
            matthews_corrcoef(y_true, y_score>.5),
            brier_score_loss(y_true, y_score)
        ]
        metric_names = ['accuracy', 'F1 Score', 'ROC AUC', 'matthews corrcoef', 'brier loss']
        
        df_data["metric_name"].extend(metric_names)
        df_data["metric_value"].extend(metrics_values)
        df_data["model_name"].extend([model_name] * len(metric_names))

    return pd.DataFrame(df_data)

def generate_roc_auc_plot(MODEL2y_pred, y_true, filename):
    """Generate and save the ROC AUC plot."""
    __fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    
    for model_name, y_score in MODEL2y_pred.items():
        fpr, tpr, __thresholds = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=model_name)
        
    ax.set_title('ROC AUC')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='best')
    
    plt.savefig(filename)

def save_metrics(df_metrics, save_path, attribute_name, folder_tables):
    """Save metrics to a CSV file and a LaTeX table."""
    csv_path = save_path / f"{attribute_name}_classification_metrics.csv"
    latex_path = folder_tables / f"{attribute_name}_comparison_classifiers.tex"
    
    df_metrics.to_csv(csv_path, index=False)
    df_metrics.to_latex(latex_path, index=False)