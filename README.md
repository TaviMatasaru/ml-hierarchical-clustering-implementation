# Hierarchical Clustering of Penguin Data

This Python script performs hierarchical clustering on a dataset containing information about penguins. The script includes data loading, preprocessing, calculation of statistical measures, conversion of categorical variables to numeric, computation of distances, and the hierarchical clustering process itself.

## Dependencies

- Python 3.8 or newer
- Pandas
- NumPy

To install the required packages, run the following command:

\`\`\`bash
pip install pandas numpy
\`\`\`

## Dataset

The script is configured to load the penguin dataset locally from:

\`\`\`
/Users/tavi/Documents/College/YEAR 3 SEM 1/ML/Hierarchical/penguins.csv
\`\`\`

Alternatively, you can uncomment the section of the code that downloads the dataset directly from the seaborn dataset repository.

## Features of the Script

1. **Data Loading**: Load the dataset either from a local file or from a URL.
2. **Preprocessing**:
   - Removal of rows containing NaN values.
   - Resetting of dataframe index after row deletion.
   - Dropping the 'species' target attribute.
3. **Statistical Calculations**:
   - Calculation of the mean and variance for numeric attributes.
4. **Conversion of Categorical to Numeric**:
   - Mapping categorical variables ('island', 'sex') to numeric values.
5. **Distance Calculations**:
   - Implementation of the Minkowski distance formula.
   - Construction of a distance matrix.
6. **Hierarchical Clustering**:
   - Functions to update the distance matrix using single, complete, and average linkage methods.
   - Calculation of dendrogram heights.
   - Execution of the agglomerative clustering algorithm.

## Usage

To run the script, navigate to the script's directory and run:

\`\`\`bash
python hierarchical_clustering.py
\`\`\`

Ensure the dataset path is correctly specified or adjust the path according to your local setup.

## Output

The script will display:
- The initial and processed number of rows in the dataset.
- Basic dataset information and first few rows of data.
- Statistical measures such as mean and variance for each numeric attribute.
- Data types of the columns after mapping.
- Final clustering results including cluster membership and dendrogram heights.

## Example of Hierarchical Clustering

An example of using the script with average linkage and targeting 30 clusters is provided at the end of the script. This can be modified to test different numbers of clusters or linkage methods.
