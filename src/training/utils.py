from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def split_into_train_valid_test(X, y, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=dev_ratio/(train_ratio+test_ratio), stratify=y_train, random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def evaluating_models_cm_and_recall(models, X_test, y_test):
    graphics_dir = 'data/graphics/evaluating'
    recall_scores = {}

    print("\nEvaluation Results:")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        recall_scores[name] = recall

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(graphics_dir, f'{name.replace(" ", "_")}_confusion_matrix.png'))
        plt.close()

def evaluating_models(models, X_test, y_test):
    graphics_dir = 'data/graphics/evaluating'
    metrics_data = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        metrics_data.append({
            'Model': name,
            'Metric': 'Recall',
            'Score': recall
        })
        metrics_data.append({
            'Model': name,
            'Metric': 'Accuracy',
            'Score': accuracy
        })
        metrics_data.append({
            'Model': name,
            'Metric': 'Precision',
            'Score': precision
        })
    
    metrics_df = pd.DataFrame(metrics_data)

    # Plotting the bar graph comparing Recall, Accuracy, and Precision
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df, palette='viridis')
    plt.title('Comparison of Model Performance: Recall, Accuracy, and Precision')
    plt.xlabel('Machine Learning Model')
    plt.ylabel('Score')
    plt.ylim(0, 1) # Scores are typically between 0 and 1
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'model_performance_comparison.png'))
    plt.close()