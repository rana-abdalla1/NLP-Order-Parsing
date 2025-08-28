# Create Datasets Package

## Overview
The Create Datasets Package is designed to generate synthetic datasets needed to train the models used in the NER Project. 

## Package Structure
```
C:.
│   README.md
│   requirements.txt
│   __init__.py
│   
├───create_datasets
│       data_loader.py
│       data_preprocessing.py    
│       data_sampler.py
│       data_statistics.py       
│       data_writer.py
│       utils.py
│       __init__.py
│
└───scripts
        main.py
```

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

```bash
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader punkt
```

## Usage
To run the project, execute the `main.py` script located in the `scripts` directory. This script orchestrates the loading, processing, and writing of data.

```bash
python scripts/main.py
```

## Classes Overview
- **DataLoader**: Responsible for loading data from Parquet files into a Pandas DataFrame and displaying its information.
- **DataSampler**: Provides methods for extracting random samples from a DataFrame.
- **DataStatistics**: Calculates and prints global statistics from the DataFrame.
- **DataPreprocessor**: Handles various data preprocessing tasks related to order extraction and cleaning.
- **DataWriter**: Manages writing cleaned data and initial source data to Parquet files and text files.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.