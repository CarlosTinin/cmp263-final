import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, accuracy_score
import matplotlib.pyplot as plt

def knn_optimizing_k(X_train, y_train, X_valid, y_valid, k_range=[3,5,7,9,11]):
    k_scores = []

    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    save_path='data/graphics/knn_performance.png'

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_valid)

        accuracy = accuracy_score(y_valid, pred_i)
        recall = recall_score(y_valid, pred_i)
        precision = precision_score(y_valid, pred_i)
        
        k_scores.append([k, accuracy, recall, precision])
        
        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracy_scores, marker='o', label='Accuracy')
    plt.plot(k_range, recall_scores, marker='o', label='Recall')
    plt.plot(k_range, precision_scores, marker='o', label='Precision')
    
    plt.title('KNN Performance for Different K Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Score')
    plt.xticks(k_range)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

    k_scores_np = np.array(k_scores)
    best_k_index = np.argmax(k_scores_np[:, 2]) # Maximize recall
    best_k = int(k_scores_np[best_k_index, 0])
    best_recall = k_scores_np[best_k_index, 2]
    
    print(f"\nBest K: {best_k}")
    print(f"Best Recall: {best_recall:.4f}")

    # Train and return the best model
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate((y_train, y_valid))
    best_knn_model = KNeighborsClassifier(n_neighbors=best_k)
    best_knn_model.fit(X_combined, y_combined)
    
    return best_knn_model, best_k

def svm_optimizing_c(X_train, y_train, X_valid, y_valid, C_range=[0.1, 1, 5]):
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    svm_scores = []
    save_path='data/graphics/svm_performance.png'
    kernel = 'linear' # Fixed kernel to 'linear'
    
    for C in C_range:
        svm = SVC(C=C, kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        pred_i = svm.predict(X_valid)

        accuracy = accuracy_score(y_valid, pred_i)
        recall = recall_score(y_valid, pred_i)
        precision = precision_score(y_valid, pred_i)
        
        svm_scores.append([C, accuracy, recall, precision])

        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)

    plt.figure(figsize=(10, 6))
    
    plt.plot(C_range, accuracy_scores, marker='o', label='Accuracy', color='skyblue')
    plt.plot(C_range, recall_scores, marker='o', label='Recall', color='lightcoral')
    plt.plot(C_range, precision_scores, marker='o', label='Precision', color='lightgreen')
    
    plt.title('SVM Performance for Different C Values (Linear Kernel)')
    plt.xlabel('C Value')
    plt.ylabel('Score')
    plt.xticks(C_range)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"SVM Plot saved to: {save_path}")
    plt.close()

    c_scores_np = np.array(svm_scores)
    best_c_index = np.argmax(c_scores_np[:, 2]) # Maximize recall
    best_c = int(c_scores_np[best_c_index, 0])
    best_recall = c_scores_np[best_c_index, 2]

    print(f"\nBest C SVM Parameter: {best_c}")
    print(f"Best Recall for SVM: {best_recall:.4f}")

    # Train and return the best model on the combined dataset
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate((y_train, y_valid))
    best_svm_model = SVC(C=best_c, kernel='linear', random_state=42)
    best_svm_model.fit(X_combined, y_combined)
    
    return best_svm_model, best_c

def gaussian_nayve_bayes(X_train, y_train, X_valid, y_valid):
    gnb = GaussianNB()
    gnb.fit(pd.concat([X_train, X_valid], ignore_index=True), np.concatenate((y_train, y_valid)))

    return gnb

def decision_tree_optimizing_max_depth(X_train, y_train, X_valid, y_valid, max_depth_range=[3, 5, 7, 9, 11]):
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    dt_scores = []
    save_path='data/graphics/dt_performance.png'

    for max_depth in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(X_train, y_train)
        pred_i = dt.predict(X_valid)

        accuracy = accuracy_score(y_valid, pred_i)
        recall = recall_score(y_valid, pred_i)
        precision = precision_score(y_valid, pred_i)

        dt_scores.append([max_depth, accuracy, recall, precision])

        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)

    plt.figure(figsize=(10, 6))
    
    plt.plot(max_depth_range, accuracy_scores, marker='o', label='Accuracy', color='skyblue')
    plt.plot(max_depth_range, recall_scores, marker='o', label='Recall', color='lightcoral')
    plt.plot(max_depth_range, precision_scores, marker='o', label='Precision', color='lightgreen')
    
    plt.title('Decision Tree Performance for Different Max Depth Values')
    plt.xlabel('Max Depth')
    plt.ylabel('Score')
    plt.xticks(max_depth_range)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Decision Tree Plot saved to: {save_path}")
    plt.close()

    dt_scores_np = np.array(dt_scores)
    best_max_depth_index = np.argmax(dt_scores_np[:, 2]) # Maximize recall
    best_max_depth = int(dt_scores_np[best_max_depth_index, 0])
    best_recall = dt_scores_np[best_max_depth_index, 2]

    print(f"\nBest Max Depth for Decision Tree: {best_max_depth}")
    print(f"Best Recall for Decision Tree: {best_recall:.4f}")

    # Train and return the best model on the combined dataset
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate((y_train, y_valid))
    best_dt_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    best_dt_model.fit(X_combined, y_combined)

    return best_dt_model, best_max_depth

def random_forest_optimizing_n_estimators(X_train, y_train, X_valid, y_valid, n_estimators_range=[10, 50, 100, 200]):
    accuracy_scores = []
    recall_scores = []
    precision_scores = []
    rf_scores = []
    save_path='data/graphics/rf_performance.png'

    for n_estimators in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=7, random_state=42)
        rf.fit(X_train, y_train)
        pred_i = rf.predict(X_valid)

        accuracy = accuracy_score(y_valid, pred_i)
        recall = recall_score(y_valid, pred_i)
        precision = precision_score(y_valid, pred_i)

        rf_scores.append([n_estimators, accuracy, recall, precision])

        accuracy_scores.append(accuracy)
        recall_scores.append(recall)
        precision_scores.append(precision)

    plt.figure(figsize=(10, 6))
    
    plt.plot(n_estimators_range, accuracy_scores, marker='o', label='Accuracy', color='skyblue')
    plt.plot(n_estimators_range, recall_scores, marker='o', label='Recall', color='lightcoral')
    plt.plot(n_estimators_range, precision_scores, marker='o', label='Precision', color='lightgreen')
    
    plt.title('Random Forest Performance for Different N Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Score')
    plt.xticks(n_estimators_range)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Random Forest Plot saved to: {save_path}")
    plt.close()

    rf_scores_np = np.array(rf_scores)
    best_n_estimators_index = np.argmax(rf_scores_np[:, 2]) # Maximize recall
    best_n_estimators = int(rf_scores_np[best_n_estimators_index, 0])
    best_recall = rf_scores_np[best_n_estimators_index, 2]

    print(f"\nBest N Estimators for Random Forest: {best_n_estimators}")
    print(f"Best Recall for Random Forest: {best_recall:.4f}")

    # Train and return the best model on the combined dataset
    X_combined = pd.concat([X_train, X_valid], ignore_index=True)
    y_combined = np.concatenate((y_train, y_valid))
    best_rf_model = RandomForestClassifier(n_estimators=best_n_estimators, random_state=42)
    best_rf_model.fit(X_combined, y_combined)
    return best_rf_model, best_n_estimators

