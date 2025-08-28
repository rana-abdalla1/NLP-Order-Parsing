def split_with_brackets(text):
    """
    Split a string while keeping the brackets as separate elements.
    """
    import re
    output_list = re.findall(r'\(|\)|\S+', text)
    return output_list

def display_dataframe_info(df):
    """
    Display the number of rows and columns in a dataframe.
    """
    print(f"Dataframe Size: {df.shape[0]} rows and {df.shape[1]} columns")
    print()