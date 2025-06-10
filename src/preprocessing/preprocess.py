from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, df):
        self.df = df
        for i in range(1, 11):
            col_name = f'A{i}_Score'
            self.df[col_name] = self.df[col_name].astype('int64')

    def drop_columns_by_index(self, indices):
        cols_to_drop = [self.df.columns[i] for i in indices]
        self.df.drop(columns=cols_to_drop, inplace=True)

    def binarize_columns(self):
        # gender column: 'f' -> 0, 'm' -> 1
        self.df['gender'] = self.df['gender'].map({'f': 0, 'm': 1})
        # jundice column: 'no' -> 0, 'yes' -> 1
        self.df['jundice'] = self.df['jundice'].map({'no': 0, 'yes': 1})
        # austim column: 'no' -> 0, 'yes' -> 1
        self.df['austim'] = self.df['austim'].map({'no': 0, 'yes': 1})
        # austim column: 'no' -> 0, 'yes' -> 1
        self.df['Class/ASD'] = self.df['Class/ASD'].map({'NO': 0, 'YES': 1})
    
    def remove_rows_with_nulls(self):
        self.df.dropna(inplace=True)

    def normalize_age_minmax(self, set):
        scaler = MinMaxScaler()
        age_data = set['age'].values.reshape(-1, 1)
        set['age'] = scaler.fit_transform(age_data)

        return set