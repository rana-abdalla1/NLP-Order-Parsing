class DataSampler:
    """
    A class to handle sampling operations on DataFrames.
    """

    @staticmethod
    def extract_sample_from_dataframe(df, n):
        """
        Extract a random sample of rows from a DataFrame.

        Parameters:
        df (DataFrame): The DataFrame to sample from.
        n (int): The number of rows to sample.

        Returns:
        DataFrame: A DataFrame containing the sampled rows.
        """
        df_random_sample = df.copy()
        
        return df_random_sample.sample(n)

    @staticmethod
    def print_dataframe(df):
        """
        Print a random sample of rows from a DataFrame.

        Parameters:
        df (DataFrame): The DataFrame to sample from.
        n (int): The number of rows to sample.
        """
        row_separator = "-" * 50
        
        for index, row in df.iterrows():
            print(f"Row {index}:")
            for col_name, value in row.items():
                print(f"  {col_name}: {value}")
            print(row_separator)
            print()
