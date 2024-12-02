import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # For file path operations
from PIL import Image  # For image handling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV


# Define the grid search parameters for each classifier
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto'],
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
    },
    'KNN': {
        'classifier__n_neighbors': [3, 5, 7],
        'classifier__weights': ['uniform', 'distance'],
    },
    'AdaBoost': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.1, 1],
    },
    'Bagging': {
        'classifier__n_estimators': [10, 20],
        'classifier__max_samples': [0.5, 1.0],
    },
    'Gradient Boosted Tree': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__max_depth': [3, 5],
    },
}

def Main():
    # Load the dataset
    file_path = 'archive/Data.csv'
    data = pd.read_csv(file_path)
    print("Dataset loaded.")

    # Load the Harder Dataset
    file_path = 'archive/extra_hard_samples.csv'
    data2 = pd.read_csv(file_path)
    print("Extra Hard Dataset loaded.")

    # Concatenate the two datasets
    data = pd.concat([data, data2], ignore_index=True)

    # Ensure 'image_name' and 'class' columns exist
    if not {'image_name', 'class'}.issubset(data.columns):
        raise ValueError("The dataset must contain 'image_name' and 'class' columns.")

    # Separate features and target variable
    X = data.drop(['image_name', 'class'], axis=1)
    y = data['class']
    image_names = data['image_name']  # Store image names for later use
    print("Features and target variable separated.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, image_train, image_test = train_test_split(
        X, y, image_names, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")

    # Define normalization functions
    normalizations = {
        'Standardization': scale_features_standard,
        #'Min-Max Scaling': scale_features_minmax  # Include if desired
    }

    # Define feature selection methods
    feature_selections = {
        'Variance Threshold': select_features_variance_threshold,
       # 'RFE': select_features_rfe,
        'SelectKBest': select_features_selectkbest
    }

    # Define PCA options
    pca_options = [2, 3, 5, 7, 10]  # Including 2 for visualization

    # Define classifiers
    classifiers = {
        'Gradient Boosted Tree': gradient_boosting_model(),
        #'Logistic Regression': logistic_regression_model(),
        'SVM': svm_model(),
        #'Random Forest': random_forest_model(),
        #'KNN': knn_model(),
        #'AdaBoost': adaboost_model(),
        #'Bagging': bagging_model()
    }

    # Initialize results list
    results = []

    # Calculate total iterations for progress tracking
    total_iterations = len(normalizations) * len(feature_selections) * len(pca_options) * len(classifiers)
    iteration = 1

    # ====================== First Pipeline: Preprocessing ======================
    print("\nStarting Preprocessing Pipeline...")
    preprocessing_results = []

    # Loop over normalization methods
    for norm_name, norm_func in normalizations.items():
        print(f"\nApplying normalization: {norm_name}")
        # Apply normalization
        X_norm = norm_func(X_train)
        X_norm_test = norm_func(X_test)
        
        # Loop over feature selection methods
        for fs_name, fs_func in feature_selections.items():
            print(f"  Applying feature selection: {fs_name}")
            if fs_name == 'RFE':
                X_fs = fs_func(X_norm, y_train)
                X_fs_test = fs_func(X_norm_test, y_test)  # Typically, y_test isn't used for feature selection
            else:
                X_fs = fs_func(X_norm, y_train)
                X_fs_test = fs_func(X_norm_test, y_test)
            
            # Store the preprocessing results
            preprocessing_results.append({
                'Normalization': norm_name,
                'Feature_Selection': fs_name,
                'X_train': X_fs,
                'X_test': X_fs_test
            })

    print("Preprocessing Pipeline completed.")

    # ====================== Second Pipeline: PCA & Classification ======================
    print("\nStarting PCA & Classification Pipeline...")

    # Define KFold cross-validation
    kf = get_kfold()

    # Iterate over each preprocessing result
    for prep in preprocessing_results:
        norm_name = prep['Normalization']
        fs_name = prep['Feature_Selection']
        X_fs = prep['X_train']
        X_fs_test = prep['X_test']
        
        # Loop over PCA options
        for n_components in pca_options:
            print(f"\nApplying PCA with {n_components} components for normalization: {norm_name}, feature selection: {fs_name}")
            
            # Fit PCA on training data
            pca, X_pca = fit_pca(X_fs, n_components)
            
            # Transform test data using the fitted PCA
            X_pca_test = pca_transform(pca, X_fs_test)
            
            print(f"  Shape after PCA: {X_pca.shape}")
            print(f"  Shape of X_pca_test: {X_pca_test.shape}")
            print("\nPlotting Elbow Curve and Silhouette Scores for KMeans Clustering...")
            plot_elbow_silhouette(X_pca_test)
            # Loop over classifiers
            for clf_name, clf in classifiers.items():
                print(f"  Training classifier: {clf_name} ({iteration}/{total_iterations})")
                iteration += 1
                
                # Create pipeline steps: PCA already applied, so only classifier is needed
                pipeline = create_pipeline([('classifier', clf)])
                
                # Get parameter grid for this classifier
                param_grid = param_grids.get(clf_name, {})
                
                if param_grid:
                    # Create GridSearchCV
                    grid_search = GridSearchCV(
                        pipeline, param_grid, cv=kf, scoring='accuracy', n_jobs=-1
                    )
                    
                    # Fit GridSearchCV on the training data
                    grid_search.fit(X_pca, y_train)
                    
                    # Get the best estimator
                    best_pipeline = grid_search.best_estimator_
                    
                    # Cross-validation score
                    mean_cv_score = grid_search.best_score_
                    best_params = grid_search.best_params_
                else:
                    # If no parameter grid is specified, just fit the pipeline
                    pipeline.fit(X_pca, y_train)
                    best_pipeline = pipeline
                    mean_cv_score = pipeline.score(X_pca, y_train)
                    best_params = {}
                
                # Test set evaluation
                y_test_pred = best_pipeline.predict(X_pca_test)
                y_test_proba = best_pipeline.predict_proba(X_pca_test)
                test_accuracy, test_f1, test_auc = evaluate_model(y_test, y_test_pred, y_test_proba)
                
                # Store results
                result = {
                    'Normalization': norm_name,
                    'Feature_Selection': fs_name,
                    'PCA_Components': n_components,
                    'Classifier': clf_name,
                    'Best_Params': best_params,
                    'CV_Mean_Accuracy': mean_cv_score,
                    'Test_Accuracy': test_accuracy,
                    'Test_F1': test_f1,
                    'Test_AUC': test_auc,
                    'X_pca_test': X_pca_test,
                    'y_test_pred': y_test_pred
                }
                results.append(result)

    print("\nPCA & Classification Pipeline completed.")

        # ====================== Identifying Top 10 Models ======================
    print("\nIdentifying Top 10 Models based on Test Accuracy...")
    results_df = pd.DataFrame(results)
    top_10_results = results_df.sort_values(by='Test_Accuracy', ascending=False).head(10)
    print("Top 10 Models:")
    print(top_10_results[['Normalization', 'Feature_Selection', 'PCA_Components', 'Classifier', 'Test_Accuracy']])

    #
    print("\nStarting Clustering Visualization for Top 10 Models...")

    # Select top 10 models
    top_10 = top_10_results.head(10).reset_index(drop=True)

    # Determine subplot layout
    num_plots = 10
    plots_per_row = 5
    num_rows = 2
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(20, 8))
    axes = axes.flatten()

    for idx, row in top_10.iterrows():
        title_base = f"Model {idx+1}: {row['Classifier']} (Norm: {row['Normalization']}, FS: {row['Feature_Selection']}, PCA: {row['PCA_Components']})"
        X_pca_test = row['X_pca_test']
        y_test_true = y_test.reset_index(drop=True)  # Ensure y_test is aligned
        y_test_pred = row['y_test_pred']
        image_test_aligned = image_test.reset_index(drop=True)  # Ensure image_test is aligned
        
        # Verify lengths
        print(f"Model {idx+1}:")
        print(f"  Length of y_test_true: {len(y_test_true)}")
        print(f"  Length of y_test_pred: {len(y_test_pred)}")
        print(f"  Length of image_test_aligned: {len(image_test_aligned)}")
        
        # Ensure all lengths match
        if not (len(y_test_true) == len(y_test_pred) == len(image_test_aligned)):
            print("  Mismatch in lengths. Skipping this model.")
            continue
        
        # Perform K-Means Clustering
        k = 5  # Example number of clusters; adjust as needed
        kmeans_labels = perform_kmeans_clustering(X_pca_test, k)
        
        # Plot K-Means Clustering
        ax = axes[idx]
        scatter = ax.scatter(X_pca_test[:, 0], X_pca_test[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
        ax.set_title(f"{title_base}\nK-Means (k={k})")
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.legend(*scatter.legend_elements(), title="Clusters")

    plt.tight_layout()
    plt.show()

    # Plot K-Means Clustering
    ax = axes[idx]
    scatter = ax.scatter(X_pca_test[:, 0], X_pca_test[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    ax.set_title(f"{title_base}\nK-Means (k={k})")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(*scatter.legend_elements(), title="Clusters")

    plt.tight_layout()
    plt.show()

    # ====================== Image Validation and Sample Display ======================
    print("\nStarting Image Validation and Sample Display...")

    # Identify the best model
    best_result = results_df.sort_values(by='Test_Accuracy', ascending=False).iloc[0]
    print("\nBest Model Parameters:")
    print(best_result[['Normalization', 'Feature_Selection', 'PCA_Components', 'Classifier', 'Test_Accuracy', 'Test_F1', 'Test_AUC']])

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
    if best_fs == 'RFE':
        X_fs = feature_selections[best_fs](X_norm, y_train)
        X_fs_test = feature_selections[best_fs](X_norm_test, y_test)  # Typically, y_test isn't used
    else:
        X_fs = feature_selections[best_fs](X_norm, y_train)
        X_fs_test = feature_selections[best_fs](X_norm_test, y_test)


    # Apply best PCA
    pca, X_pca = fit_pca(X_fs, best_pca_components)
    X_pca_test = pca_transform(pca, X_fs_test)


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

    # ====================== Image Validation and Sample Display ======================
    print("\nStarting Image Validation and Sample Display...")

    # Collect predictions from the best model
    best_predictions = y_test_pred  # NumPy array
    best_true_labels = y_test
    best_image_names = image_test.reset_index(drop=True)  # Reset index to align with predictions

    # Convert best_predictions to a Pandas Series and reset index
    best_predictions = pd.Series(best_predictions).reset_index(drop=True)

    # Ensure best_true_labels is a Series with reset index
    best_true_labels = best_true_labels.reset_index(drop=True)

    # Create a DataFrame for analysis
    validation_df = pd.DataFrame({
        'image_name': best_image_names,
        'true_label': best_true_labels,
        'predicted_label': best_predictions
    })

    # Verify the DataFrame
    print("\nValidation DataFrame created successfully.")
    print(validation_df.head())
    print(f"Shape of validation_df: {validation_df.shape}")  # Should be (1334, 3)
    # Identify correct and incorrect predictions
    correct_predictions = validation_df[validation_df['true_label'] == validation_df['predicted_label']]
    incorrect_predictions = validation_df[validation_df['true_label'] != validation_df['predicted_label']]

    # Select top 5 correct and top 5 incorrect samples
    top_correct = correct_predictions.head(5)
    top_incorrect = incorrect_predictions.head(5)

    # Display top correct predictions
    display_image_samples(top_correct, "Top Correct")

    # Display top incorrect predictions
    display_image_samples(top_incorrect, "Top Incorrect")
#Function to plot elbow curve and silhouette scores
# Function to plot elbow curve and silhouette scores
def plot_elbow_silhouette(X):
    inertia = []
    silhouette_scores = []
    K = range(2, 11)  # Vary k from 2 to 10
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        inertia.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"  k={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
    
    # Plot the elbow curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    
    # Plot the silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores For Various k')
    
    plt.tight_layout()
    plt.show()
# Function to display images in a grid
def display_image_samples(samples_df, title_prefix):
    num_samples = samples_df.shape[0]
    cols = 5
    rows = num_samples // cols + int(num_samples % cols != 0)
    plt.figure(figsize=(20, 4 * rows))
    
    # Use a separate counter starting at 1
    for i, (idx, row) in enumerate(samples_df.iterrows(), 1):
        image_path = get_image_path(row['image_name'], row['true_label'])
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        img = Image.open(image_path)
        plt.subplot(rows, cols, i)  # Use the counter 'i' instead of 'idx + 1'
        plt.imshow(img, cmap='gray')  # Adjust cmap as needed
        plt.title(f"Name: {row['image_name']}\nTrue: {row['true_label']}, Pred: {row['predicted_label']}")
        plt.axis('off')
    
    plt.suptitle(f"{title_prefix} Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# Function to get the image path based on the image name and class label
def get_image_path(image_name, class_label):
    base_dir = 'archive/images'
    class_dir = class_label.lower()  # Ensure directory names are lowercase
    image_path = os.path.join(base_dir, class_dir, image_name)
    return image_path
# Function to scale features using standardization
def scale_features_standard(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
# Function to scale features using Min-Max scaling
def scale_features_minmax(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
# Function to perform PCA transformation
def pca_transform(pca, X):
    X_pca = pca.transform(X)
    return X_pca
# Function to fit PCA
def fit_pca(X, n_components):
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X)
    return pca, X_pca
# Function to select features using variance threshold
def select_features_variance_threshold(X, y, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    X_new = selector.fit_transform(X)
    return X_new
# Function to select features using Recursive Feature Elimination (RFE)
def select_features_rfe(X, y, n_features=50):
    estimator = RandomForestClassifier(random_state=0)
    selector = RFE(estimator, n_features_to_select=n_features, step=10)
    selector.fit(X, y)
    X_new = selector.transform(X)
    return X_new
# Function to select features using SelectKBest
def select_features_selectkbest(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new
# Function to create a Logistic Regression model
def logistic_regression_model():
    return LogisticRegression(max_iter=1000, random_state=0)
# Function to create a Random Forest model
def random_forest_model():
    return RandomForestClassifier(random_state=0)
# Function to create an SVM model
def svm_model():
    return SVC(probability=True, random_state=0)  # Enable predict_proba
# Function to create a KNN model
def knn_model():
    return KNeighborsClassifier()
# Function to create an AdaBoost model
def adaboost_model():
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
        n_estimators=50,
        random_state=0
    )
# Function to create a Bagging model
def bagging_model():
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        n_estimators=10,
        random_state=0
    )
# Function to create a Gradient Boosted Tree model
def gradient_boosting_model():
    return GradientBoostingClassifier(random_state=0)
# Function to get KFold cross-validation
def get_kfold():
    return KFold(n_splits=4, shuffle=True, random_state=0)
# Function to create a pipeline
def create_pipeline(steps):
    return Pipeline(steps)
# Function to evaluate a model
def evaluate_model(y_true, y_pred, y_proba, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    # Binarize the output for ROC AUC
    classes = sorted(list(set(y_true)))
    y_true_bin = label_binarize(y_true, classes=classes)
    if y_true_bin.shape[1] == 1:
        # Binary classification
        auc = roc_auc_score(y_true_bin, y_proba[:,1])
    else:
        # Multiclass classification
        auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average=average)
    return accuracy, f1, auc
# Function to perform K-Means clustering
def perform_kmeans_clustering(X, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"    K-Means Silhouette Score: {silhouette_avg:.4f}")
    return labels
# Function to perform Agglomerative clustering
def perform_agglomerative_clustering(X, k):
    agglo = AgglomerativeClustering(n_clusters=k)
    labels = agglo.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    print(f"    Agglomerative Clustering Silhouette Score: {silhouette_avg:.4f}")
    return labels
# Function to plot clusters
def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(
            X[labels == label, 0],
            X[labels == label, 1],
            label=f'Cluster {label}'
        )
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.show()
# Function to plot a metric by a parameter
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
# Function to plot a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_true)))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {classifier_name}')
    plt.show()

if __name__ == '__main__':
    Main()
