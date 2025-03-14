import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
import os

def create_classification_visualizations(X, y, output_dir='visualizations'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert AQI to categories for classification
    def aqi_to_category(aqi):
        if aqi <= 50: return 'Good'
        elif aqi <= 100: return 'Moderate'
        elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
        elif aqi <= 200: return 'Unhealthy'
        elif aqi <= 300: return 'Very Unhealthy'
        else: return 'Hazardous'
    
    y_cat = pd.Series(y).apply(aqi_to_category)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_cat)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store models and their predictions
    models = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    models['Logistic Regression'] = (lr, y_pred_lr)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{output_dir}/logistic_regression_cm.png')
    plt.close()
    
    # 2. Decision Tree Classification
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred_dt = dt.predict(X_test_scaled)
    models['Decision Tree'] = (dt, y_pred_dt)
    
    # Feature importance for Decision Tree
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Decision Tree Feature Importance')
    plt.savefig(f'{output_dir}/decision_tree_classification_importance.png')
    plt.close()
    
    # 3. Random Forest Classification
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    models['Random Forest'] = (rf, y_pred_rf)
    
    # Feature importance for Random Forest
    plt.figure(figsize=(12, 6))
    feature_importance_rf = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(data=feature_importance_rf, x='importance', y='feature')
    plt.title('Random Forest Feature Importance')
    plt.savefig(f'{output_dir}/random_forest_classification_importance.png')
    plt.close()
    
    # 4. K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    models['KNN'] = (knn, y_pred_knn)
    
    # Create 2D visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Train a new KNN model on PCA data for visualization
    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    
    # Create mesh grid for decision boundary
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    # Plot KNN decision boundary
    plt.figure(figsize=(10, 8))
    Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, alpha=0.8)
    plt.colorbar(scatter)
    plt.title('KNN Decision Boundary (PCA visualization)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(f'{output_dir}/knn_decision_boundary.png')
    plt.close()
    
    # 5. Support Vector Machines
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    models['SVM'] = (svm, y_pred_svm)
    
    # 6. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    y_pred_nb = nb.predict(X_test_scaled)
    models['Naive Bayes'] = (nb, y_pred_nb)
    
    # Model Comparison
    accuracies = {name: accuracy_score(y_test, pred) for name, (_, pred) in models.items()}
    
    plt.figure(figsize=(12, 6))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.title('Classification Models - Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_model_comparison.png')
    plt.close()
    
    # Save classification reports
    with open(f'{output_dir}/classification_reports.txt', 'w') as f:
        for name, (_, y_pred) in models.items():
            f.write(f'\n{name} Classification Report:\n')
            f.write(classification_report(y_test, y_pred))
            f.write('\n' + '='*50 + '\n')
    
    return models, accuracies, le.classes_
