class DataLoader:
    """
    A class to load a Parquet file into a Pandas DataFrame.
    """
    def __init__(self, parquet_file_path):
        self.parquet_file_path = parquet_file_path
        self.df = None

    def load_parquet_to_dataframe(self):
        import pyarrow.parquet as pq
        import pandas as pd

        # Read the Parquet file into a DataFrame
        table = pq.read_table(self.parquet_file_path)
        self.df = table.to_pandas()

        # Display the size of the DataFrame
        self.display_dataframe_info()
        return self.df

    def display_dataframe_info(self):
        if self.df is not None:
            print(f"Dataframe Size: {self.df.shape[0]} rows and {self.df.shape[1]} columns")