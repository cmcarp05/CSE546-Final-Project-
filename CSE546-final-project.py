import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def Main():
    # Load the dataset
    file_path = 'archive/Data.csv'
    data = pd.read_csv(file_path)
    print("Dataset loaded.")

    #Load the Harder Dataset
    file_path = 'archive/extra_hard_samples.csv'
    data2 = pd.read_csv(file_path)
    print("Extra Hard Dataset loaded.")

    #Concatenate the two datasets
    data = pd.concat([data, data2], ignore_index=True)

    # Separate features and target variable
    X = data.drop(['image_name', 'class'], axis=1)
    y = data['class']
    print("Features and target variable separated.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")

    # Define normalization functions
    normalizations = {
        'Standardization': scale_features_standard,
        #'Min-Max Scaling': scale_features_minmax
    }

    # Define PCA options
    pca_options = [3, 5, 7, 10]

    # Define feature selection methods
    feature_selections = {
        'Univariate Selection': select_features_univariate,
        'RFE': select_features_rfe
    }

    # Define classifiers
    classifiers = {
        'Logistic Regression': logistic_regression_model(),
        'SVM': svm_model(),
        'Random Forest': random_forest_model(),
        #'KNN': knn_model(),
        #'AdaBoost': adaboost_model(),
        #'Bagging': bagging_model()
    }

    # Initialize results list
    results = []

    total_iterations = len(normalizations) * len(feature_selections) * len(pca_options) * len(classifiers)
    iteration = 1

    # Loop over normalization methods
    for norm_name, norm_func in normalizations.items():
        print(f"\nApplying normalization: {norm_name}")
        # Apply normalization
        X_norm = norm_func(X_train)
        X_norm_test = norm_func(X_test)
        
        # Loop over feature selection methods
        for fs_name, fs_func in feature_selections.items():
            print(f"  Applying feature selection: {fs_name}")
            X_fs = fs_func(X_norm, y_train)
            X_fs_test = fs_func(X_norm_test, y_test)
            
            # Loop over PCA options
            for n_components in pca_options:
                print(f"    Applying PCA with {n_components} components")
                X_pca = pca_transform(X_fs, n_components)
                X_pca_test = pca_transform(X_fs_test, n_components)
                
                # Loop over classifiers
                for clf_name, clf in classifiers.items():
                    print(f"      Training classifier: {clf_name} ({iteration}/{total_iterations})")
                    iteration += 1
                    # Create pipeline
                    steps = [('classifier', clf)]
                    pipeline = create_pipeline(steps)
                    
                    # Cross-validation
                    skf = get_stratified_kfold()
                    cv_scores = []
                    
                    fold_number = 1
                    for train_index, val_index in skf.split(X_pca, y_train):
                        print(f"        Cross-validation fold {fold_number}/4")
                        fold_number += 1
                        X_fold_train, X_fold_val = X_pca[train_index], X_pca[val_index]
                        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
                        
                        pipeline.fit(X_fold_train, y_fold_train)
                        y_pred = pipeline.predict(X_fold_val)
                        y_proba = pipeline.predict_proba(X_fold_val)
                        
                        accuracy, f1, auc = evaluate_model(y_fold_val, y_pred, y_proba)
                        cv_scores.append((accuracy, f1, auc))
                    
                    # Average CV scores
                    avg_scores = pd.DataFrame(cv_scores, columns=['Accuracy', 'F1', 'AUC']).mean()
                    
                    # Test set evaluation
                    pipeline.fit(X_pca, y_train)
                    y_test_pred = pipeline.predict(X_pca_test)
                    y_test_proba = pipeline.predict_proba(X_pca_test)
                    test_accuracy, test_f1, test_auc = evaluate_model(y_test, y_test_pred, y_test_proba)
                    
                    # Store results
                    result = {
                        'Normalization': norm_name,
                        'Feature_Selection': fs_name,
                        'PCA_Components': n_components,
                        'Classifier': clf_name,
                        'CV_Accuracy': avg_scores['Accuracy'],
                        'CV_F1': avg_scores['F1'],
                        'CV_AUC': avg_scores['AUC'],
                        'Test_Accuracy': test_accuracy,
                        'Test_F1': test_f1,
                        'Test_AUC': test_auc
                    }
                    results.append(result)
    # Convert results to DataFrame and display
    results_df = pd.DataFrame(results)
    print("\nAll iterations completed.")
    print(results_df.sort_values(by='Test_Accuracy', ascending=False))

    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print("Results saved to 'model_results.csv'.")

    # Plotting results
    plot_metric_by_parameter(results_df, 'Test_Accuracy', 'Classifier')
    plot_metric_by_parameter(results_df, 'Test_Accuracy', 'Normalization')
    plot_metric_by_parameter(results_df, 'Test_Accuracy', 'Feature_Selection')
    plot_metric_by_parameter(results_df, 'Test_Accuracy', 'PCA_Components')

    # Display top results
    display_top_results(results_df, 'Test_Accuracy')

    # Assuming the best model is the first in the sorted DataFrame
    best_result = results_df.sort_values(by='Test_Accuracy', ascending=False).iloc[0]
    print("\nBest Model Parameters:")
    print(best_result)

    # Re-train the best model on the entire training set and evaluate on the test set
    best_norm = best_result['Normalization']
    best_fs = best_result['Feature_Selection']
    best_pca_components = best_result['PCA_Components']
    best_classifier_name = best_result['Classifier']
    best_classifier = classifiers[best_classifier_name]

    # Apply best normalization
    X_norm = normalizations[best_norm](X_train)
    X_norm_test = normalizations[best_norm](X_test)

    # Apply best feature selection
    X_fs = feature_selections[best_fs](X_norm, y_train)
    X_fs_test = feature_selections[best_fs](X_norm_test, y_test)

    # Apply best PCA
    X_pca = pca_transform(X_fs, best_pca_components)
    X_pca_test = pca_transform(X_fs_test, best_pca_components)

    # Train best model
    best_pipeline = create_pipeline([('classifier', best_classifier)])
    best_pipeline.fit(X_pca, y_train)
    y_test_pred = best_pipeline.predict(X_pca_test)
    y_test_proba = best_pipeline.predict_proba(X_pca_test)

    # Evaluate best model
    test_accuracy, test_f1, test_auc = evaluate_model(y_test, y_test_pred, y_test_proba)
    print(f"\nBest Model Test Performance:\nAccuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")

    # Plot confusion matrix for best model
    plot_confusion_matrix(y_test, y_test_pred, best_classifier_name)

# Function to scale the features using StandardScaler
def scale_features_standard(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Function to scale the features using MinMaxScaler
def scale_features_minmax(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# PCA Function
def pca_transform(X, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

# Feature Selection using RFE
def select_features_rfe(X, y, n_features=50):
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features_to_select=n_features, step=10)
    X_new = selector.fit_transform(X, y)
    return X_new

# Feature Selection using Univariate Selection
def select_features_univariate(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

# Classifier Logistic Regression
def logistic_regression_model():
    return LogisticRegression(max_iter=1000)

# Classifier Random Forest
def random_forest_model():
    return RandomForestClassifier()

# Classifier SVM
def svm_model():
    return SVC(probability=True)  # Set probability=True to enable predict_proba

# Classifier KNN
def knn_model():
    return KNeighborsClassifier()

# Ensemble Classifier AdaBoost
def adaboost_model():
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=0
    )

# Ensemble Classifier Bagging
def bagging_model():
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=0
    )

# Function to get Stratified KFold
def get_stratified_kfold():
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

# Function to create a pipeline
def create_pipeline(steps):
    return Pipeline(steps)

# Function to evaluate the model
def evaluate_model(y_true, y_pred, y_proba, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    # Binarize the output for ROC AUC
    classes = sorted(list(set(y_true)))
    y_true_bin = label_binarize(y_true, classes=classes)
    auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average=average)
    return accuracy, f1, auc

# Function to plot metrics by parameter
def plot_metric_by_parameter(results_df, metric, parameter):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=parameter, y=metric, data=results_df)
    plt.title(f'{metric} by {parameter}')
    plt.ylabel(metric)
    plt.xlabel(parameter)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to display top results
def display_top_results(results_df, metric, top_n=5):
    sorted_df = results_df.sort_values(by=metric, ascending=False).head(top_n)
    print(f"\nTop {top_n} results based on {metric}:")
    print(sorted_df)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_true)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {classifier_name}')
    plt.show()

if __name__ == '__main__':
    Main()