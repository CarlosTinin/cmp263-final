from visualization.explore import DataExplorer
from preprocessing.preprocess import Preprocessor
from training.models import knn_optimizing_k, svm_optimizing_c, gaussian_nayve_bayes, decision_tree_optimizing_max_depth, random_forest_optimizing_n_estimators
from training.utils import split_into_train_valid_test, evaluating_models

################################################################
#            1. Data visualization and exploration             #
################################################################

explorer = DataExplorer('data/Autism-Child-Data.arff')
print("Data Explorer initialized.")
explorer.load_data() # Load the dataset from Weka ARFF file
explorer.show_head_and_tail() # Print summary of the dataset, first and last 5 rows
#explorer.show_info() # Print dataset shape and column types
#explorer.show_unique_values(19) # column 'relation'
#explorer.show_unique_values(18) # column 'age_desc'
#explorer.show_unique_values(12) # column 'ethnicity'
#explorer.show_unique_values(20) # column 'result' (target attr)
#explorer.plot_target_histogram() # Plot histogram of target attribute 'Class/ASD'
print("Data Explorer finished.")

################################################################
#        2. Preprocessing (cleaning and codification)          #
################################################################

preprocessor = Preprocessor(explorer.df)
preprocessor.drop_columns_by_index([12, 15, 16, 17, 18, 19]) # Drop columns 'relation', 'age_desc', 'ethnicity', 'result', 'used_app_before' by index
preprocessor.binarize_columns()
preprocessor.remove_rows_with_nulls()
#explorer.show_head_and_tail() # Print summary again to see the changes

################################################################
#                       3. Data split                          #
################################################################

X = explorer.df.drop(['Class/ASD'], axis=1)
y = explorer.df['Class/ASD'].values

X_train, X_valid, X_test, y_train, y_valid, y_test = split_into_train_valid_test(X, y)

#print(y_valid.shape)
#print(X_test.shape)
#print(X_valid.shape)

################################################################
#               4. Preprocessing (normalization)               #
################################################################

X_train = preprocessor.normalize_age_minmax(X_train)
X_valid = preprocessor.normalize_age_minmax(X_valid)
X_test = preprocessor.normalize_age_minmax(X_test)
#print(X_train.head(10)) # Print first 10 rows to check normalization

################################################################
#                         5. Training                          #
################################################################

# kNN
best_knn_model, best_k = knn_optimizing_k(X_train, y_train, X_valid, y_valid)

# SVM
best_svm_model, best_c = svm_optimizing_c(X_train, y_train, X_valid, y_valid)

# GNB
gnb = gaussian_nayve_bayes(X_train, y_train, X_valid, y_valid)

# DCT
dct, best_max_depth = decision_tree_optimizing_max_depth(X_train, y_train, X_valid, y_valid)

# Random Forest
rmf, best_estimators_number = random_forest_optimizing_n_estimators(X_train, y_train, X_valid, y_valid)


################################################################
#                       6. Evaluation                          #
################################################################

models = {
    'KNN': best_knn_model,
    'SVM': best_svm_model,
    'GaussianNB': gnb,
    'DecisionTree': dct,
    'RandomForest': rmf
}

evaluating_models(models, X_test, y_test)

################################################################
#                 7. Final training (To Do)                    #
################################################################

# train the best svm with whole dataset