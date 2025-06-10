import pandas as pd
import arff
import matplotlib.pyplot as plt
import os

class DataExplorer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        try:
            with open(self.file_path, 'r') as f:
                dataset = arff.load(f)
            self.df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
            print(f"Data loaded from: {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def show_head_and_tail(self, n=5):
        print(f"\nFirst {n} rows:")
        print(self.df.head(n))

        print(f"\nLast {n} rows:")
        print(self.df.tail(n))

    def show_info(self):
        print("\nDataset shape (rows, columns):")
        print(self.df.shape)

        print("\nColumn names and types:")
        print(self.df.dtypes)
    
    def show_unique_values(self, column_index):
        try:
            column_name = self.df.columns[column_index]
            print(f"\nUnique values in column [{column_index}] '{column_name}':")
            print(self.df[column_name].value_counts(dropna=False))
            
            none_count = self.df[column_name].isnull().sum()
            print(f"\nNone values in column '{column_name}': {none_count}")
        except Exception as e:
            print(f"Error: {e}")
    
    def plot_target_histogram(self, column='Class/ASD', output_dir='data/graphics'):
        plt.figure(figsize=(8, 6))
        self.df[column].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Distribution of "{column}"')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save to file
        output_path = os.path.join(output_dir, f'target_attr_histogram.png')
        plt.savefig(output_path)
        plt.close()