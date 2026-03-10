import sys  # Import sys for program exit on critical errors
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for tabular data processing
from pathlib import Path  # Import Path for convenient file path handling
from scipy.stats import mannwhitneyu, spearmanr  # Import Mann-Whitney U test and Spearman correlation
from sklearn.impute import SimpleImputer  # Import imputer for missing value filling
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder  # Import discretizer and label encoder
from sklearn.metrics import mutual_info_score  # Import mutual information calculation function

INPUT_CSV = Path("Total.csv")  # Input CSV file path
OUTPUT_DIR = Path("feature_selection_results")  # Output directory path
OUTPUT_DIR.mkdir(exist_ok=True)  # Create output directory automatically if it does not exist

LABEL_COL_INDEX = 1  # Label column index, using Python zero-based indexing, so column 2 is index 1
FEATURE_START_COL_INDEX = 2  # Feature start column index, so features begin from column 3
ENCODING = "utf-8-sig"  # Encoding for CSV output, compatible with Excel and non-ASCII text
MWU_P_THRESHOLD = 0.05  # Significance threshold for Mann-Whitney U test
SPEARMAN_THRESHOLD = 0.90  # Spearman correlation threshold for redundancy removal
TOP_K = 50  # Number of final features to select with mRMR
MRMR_CRITERION = "MID"  # mRMR criterion, choose either "MID" or "MIQ"
N_BINS = 10  # Number of quantile bins used before mRMR
RANDOM_STATE = 42  # Random seed for reproducibility
EPS = 1e-12  # Small constant to avoid division by zero in MIQ

def discrete_mi(x_disc: np.ndarray, y_disc: np.ndarray) -> float:  # Define a function to compute mutual information between two discrete arrays
    return float(mutual_info_score(x_disc, y_disc))  # Return mutual information as a float

def load_and_prepare_data(path: Path):  # Define a function to load and preprocess the dataset
    if not path.exists():  # Check whether the input file exists
        sys.exit(f"[ERROR] File not found: {path}")  # Exit the program if the file does not exist

    df = pd.read_csv(path, encoding=ENCODING)  # Read the CSV file into a DataFrame
    if df.shape[1] <= FEATURE_START_COL_INDEX:  # Ensure there are enough columns for ID, label, and at least one feature
        sys.exit("[ERROR] The dataset must contain at least: ID column + label column + one feature column.")  # Exit if the dataset structure is invalid

    df.iloc[:, LABEL_COL_INDEX] = pd.to_numeric(df.iloc[:, LABEL_COL_INDEX], errors="coerce")  # Convert label column to numeric, coercing invalid values to NaN
    X_all = df.iloc[:, FEATURE_START_COL_INDEX:].apply(pd.to_numeric, errors="coerce")  # Convert all feature columns to numeric, coercing invalid values to NaN

    df.iloc[:, LABEL_COL_INDEX] = df.iloc[:, LABEL_COL_INDEX].fillna(df.iloc[:, LABEL_COL_INDEX].mean())  # Fill missing values in the label column using the mean
    X_all = X_all.fillna(X_all.mean())  # Fill missing values in each feature column using the column mean

    y = df.iloc[:, LABEL_COL_INDEX].astype(int)  # Convert the label column to integer type for binary classification
    X = X_all.copy()  # Copy the full feature matrix

    print(f"Sample size: {len(df)}, Original number of features: {X.shape[1]}")  # Print sample size and original feature count

    unique_y = sorted(y.unique().tolist())  # Get the sorted unique values in the label vector
    if len(unique_y) != 2:  # Check whether the task is binary classification
        sys.exit(f"[ERROR] The current label column is not binary. Detected classes: {unique_y}. Mann-Whitney U test is intended for binary classification.")  # Exit if not binary

    return df, X, y  # Return the original DataFrame, feature matrix, and label vector

def mann_whitney_filter(X: pd.DataFrame, y: pd.Series):  # Define a function for Mann-Whitney U feature filtering
    p_values = {}  # Create a dictionary to store p-values for each feature

    class_values = sorted(y.unique())  # Get the two class labels in sorted order
    class0 = class_values[0]  # First class label
    class1 = class_values[1]  # Second class label

    for col in X.columns:  # Iterate through each feature column
        grp0 = X.loc[y == class0, col]  # Extract feature values for class 0
        grp1 = X.loc[y == class1, col]  # Extract feature values for class 1

        try:  # Try to perform the statistical test
            _, p = mannwhitneyu(grp1, grp0, alternative="two-sided")  # Perform the two-sided Mann-Whitney U test
        except Exception:  # If any exception occurs for this feature
            p = 1.0  # Assign a non-significant p-value

        p_values[col] = p  # Save the p-value for the current feature

    p_series = pd.Series(p_values).sort_values()  # Convert the dictionary to a Series and sort by p-value
    sig_series = p_series[p_series < MWU_P_THRESHOLD]  # Keep only significant features below the threshold
    selected_features = sig_series.index.tolist()  # Extract the names of significant features
    X_mwu = X[selected_features].copy()  # Build a new feature matrix using significant features only

    p_series.to_csv(OUTPUT_DIR / "01_MWU_p_values.csv", header=["p_value"], encoding=ENCODING)  # Save all p-values to CSV
    sig_series.to_csv(OUTPUT_DIR / "02_MWU_significant_features.csv", header=["p_value"], encoding=ENCODING)  # Save significant features to CSV

    print(f"Number of features retained after Mann-Whitney U test: {X_mwu.shape[1]}")  # Print retained feature count after MWU

    return X_mwu, p_series, sig_series  # Return filtered features, all p-values, and significant p-values

def spearman_reduction(X: pd.DataFrame, p_series: pd.Series):  # Define a function for Spearman correlation-based redundancy removal
    if X.shape[1] <= 1:  # If there is only one or zero features, redundancy removal is unnecessary
        print("Spearman redundancy removal skipped: number of features <= 1")  # Print informational message
        return X.copy(), [], pd.DataFrame()  # Return the input features and empty outputs

    corr_matrix = X.corr(method="spearman").abs()  # Compute the absolute Spearman correlation matrix
    cols = corr_matrix.columns.tolist()  # Get feature names as a list
    to_drop = set()  # Create a set to record redundant features to remove
    drop_records = []  # Create a list to record detailed removal information

    for i in range(len(cols)):  # Loop over all feature indices
        for j in range(i + 1, len(cols)):  # Compare each pair only once
            col_i = cols[i]  # Name of the first feature
            col_j = cols[j]  # Name of the second feature
            corr_val = corr_matrix.loc[col_i, col_j]  # Absolute Spearman correlation between the two features

            if corr_val > SPEARMAN_THRESHOLD:  # If correlation exceeds the redundancy threshold
                p_i = p_series.get(col_i, 1.0)  # Get MWU p-value for the first feature
                p_j = p_series.get(col_j, 1.0)  # Get MWU p-value for the second feature

                if p_i <= p_j:  # Keep the feature with the smaller MWU p-value
                    drop_feature = col_j  # Drop the second feature
                    keep_feature = col_i  # Keep the first feature
                else:  # Otherwise
                    drop_feature = col_i  # Drop the first feature
                    keep_feature = col_j  # Keep the second feature

                if drop_feature not in to_drop:  # Avoid duplicate recording
                    to_drop.add(drop_feature)  # Add redundant feature to the drop set
                    drop_records.append([keep_feature, drop_feature, corr_val, p_series.get(keep_feature, np.nan), p_series.get(drop_feature, np.nan)])  # Save redundancy removal details

    X_reduced = X.drop(columns=list(to_drop), errors="ignore").copy()  # Remove redundant features from the matrix

    drop_df = pd.DataFrame(drop_records, columns=["kept_feature", "dropped_feature", "abs_spearman_corr", "kept_feature_mwu_p", "dropped_feature_mwu_p"])  # Convert drop records to a DataFrame
    corr_matrix.to_csv(OUTPUT_DIR / "03_Spearman_correlation_matrix.csv", encoding=ENCODING)  # Save the Spearman correlation matrix
    drop_df.to_csv(OUTPUT_DIR / "04_Spearman_dropped_features.csv", index=False, encoding=ENCODING)  # Save dropped feature records
    X_reduced.to_csv(OUTPUT_DIR / "05_Features_after_Spearman.csv", index=False, encoding=ENCODING)  # Save features after redundancy removal

    print(f"Number of features retained after Spearman redundancy removal: {X_reduced.shape[1]}")  # Print remaining feature count after Spearman filtering

    return X_reduced, drop_records, corr_matrix  # Return reduced features, drop records, and correlation matrix

def mrmr_select(X: pd.DataFrame, y: pd.Series, top_k: int = TOP_K, criterion: str = MRMR_CRITERION):  # Define a function for mRMR feature selection
    if X.shape[1] == 0:  # Check whether there are any input features
        print("mRMR skipped: no input features available")  # Print informational message
        return [], pd.DataFrame(), pd.DataFrame()  # Return empty outputs

    feat_names = X.columns.tolist()  # Get the list of feature names

    y_enc = LabelEncoder().fit_transform(y.astype(str).values)  # Encode class labels as integers

    imp = SimpleImputer(strategy="median")  # Create a median imputer
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feat_names, index=X.index)  # Impute missing values using the median

    actual_bins = min(N_BINS, max(2, int(X_imp.shape[0] // 5)))  # Dynamically adjust the number of bins according to sample size
    disc = KBinsDiscretizer(n_bins=actual_bins, encode="ordinal", strategy="quantile")  # Create a quantile-based discretizer
    X_disc = pd.DataFrame(disc.fit_transform(X_imp), columns=feat_names, index=X.index).astype(int)  # Discretize features into ordinal integers

    relevance = {}  # Create a dictionary to store feature relevance to the label
    for f in feat_names:  # Iterate over all features
        relevance[f] = discrete_mi(X_disc[f].values, y_enc)  # Compute MI between the feature and the label

    rel_df = pd.DataFrame({"feature": feat_names, "relevance_MI_to_y": [relevance[f] for f in feat_names]})  # Build a relevance DataFrame
    rel_df = rel_df.sort_values("relevance_MI_to_y", ascending=False)  # Sort features by relevance in descending order
    rel_df.to_csv(OUTPUT_DIR / "06_mRMR_relevance.csv", index=False, encoding=ENCODING)  # Save feature-label relevance scores

    k = int(min(max(1, top_k), len(feat_names)))  # Ensure top_k is at least 1 and not larger than the number of available features
    selected = []  # Initialize the selected feature list
    remaining = set(feat_names)  # Initialize the remaining candidate feature set
    selection_records = []  # Initialize a list to store the step-by-step selection process

    def avg_redundancy(f, selected_list):  # Define an inner function to compute average redundancy for a candidate feature
        if not selected_list:  # If no feature has been selected yet
            return 0.0  # Redundancy is zero
        f_vec = X_disc[f].values  # Get discretized values of the candidate feature
        r_vals = []  # Create a list to store redundancy values
        for s in selected_list:  # Loop over selected features
            s_vec = X_disc[s].values  # Get discretized values of the selected feature
            r_vals.append(discrete_mi(f_vec, s_vec))  # Compute MI between the candidate and selected feature
        return float(np.mean(r_vals))  # Return the average redundancy

    for step in range(k):  # Perform greedy selection for top-k features
        best_f = None  # Initialize best feature for the current step
        best_score = -np.inf  # Initialize best score as negative infinity
        best_D = None  # Initialize best relevance
        best_R = None  # Initialize best redundancy

        for f in remaining:  # Loop over all remaining candidate features
            D = relevance[f]  # Get feature relevance to the label
            R = avg_redundancy(f, selected)  # Compute average redundancy to already selected features

            if criterion.upper() == "MIQ":  # If MIQ criterion is selected
                score = D / (R + EPS)  # Compute MIQ score
            else:  # Otherwise use MID by default
                score = D - R  # Compute MID score

            if score > best_score:  # If this candidate has a better score
                best_score = score  # Update best score
                best_f = f  # Update best feature
                best_D = D  # Update best relevance
                best_R = R  # Update best redundancy

        if best_f is None:  # If no valid feature is found
            break  # Stop the selection process early

        selected.append(best_f)  # Add the best feature to the selected list
        remaining.remove(best_f)  # Remove it from the candidate set
        selection_records.append([step + 1, best_f, best_D, best_R, best_score])  # Record the current selection step

    with open(OUTPUT_DIR / "07_mRMR_selected_features.txt", "w", encoding=ENCODING) as f:  # Open the text file for writing selected feature names
        for fea in selected:  # Iterate through selected features
            f.write(f"{fea}\n")  # Write one feature name per line

    selection_df = pd.DataFrame(selection_records, columns=["rank", "feature", "relevance_D", "redundancy_R", "score"])  # Build a DataFrame for the selection process
    selection_df.to_csv(OUTPUT_DIR / "08_mRMR_selection_process.csv", index=False, encoding=ENCODING)  # Save the mRMR step-by-step process
    X_selected = X_imp[selected].copy()  # Extract the final selected feature matrix
    X_selected.to_csv(OUTPUT_DIR / "09_mRMR_selected_feature_values.csv", index=False, encoding=ENCODING)  # Save the selected feature values

    print(f"Number of features finally selected by mRMR ({criterion.upper()}): {len(selected)}")  # Print the final number of selected features

    return selected, rel_df, selection_df  # Return selected feature names, relevance table, and selection process table

def save_final_dataset(df: pd.DataFrame, y: pd.Series, X_final: pd.DataFrame):  # Define a function to save the final modeling dataset
    id_part = df.iloc[:, :FEATURE_START_COL_INDEX].copy()  # Keep the first two columns, usually ID and label
    final_df = pd.concat([id_part, X_final], axis=1)  # Concatenate ID/label with final selected features
    final_df.to_csv(OUTPUT_DIR / "10_Final_selected_dataset.csv", index=False, encoding=ENCODING)  # Save the final dataset
    print(f"Final modeling dataset saved, shape: {final_df.shape}")  # Print the shape of the final dataset

def main():  # Define the main workflow function
    df, X, y = load_and_prepare_data(INPUT_CSV)  # Load and preprocess the input data

    X_mwu, p_series, sig_series = mann_whitney_filter(X, y)  # Step 1: perform Mann-Whitney U filtering
    if X_mwu.shape[1] == 0:  # Check whether any features remain after MWU
        sys.exit("[ERROR] No features were retained after the Mann-Whitney U test. Please check the data or relax the p-value threshold.")  # Exit if none remain

    X_spearman, drop_records, corr_matrix = spearman_reduction(X_mwu, p_series)  # Step 2: perform Spearman redundancy removal
    if X_spearman.shape[1] == 0:  # Check whether any features remain after Spearman filtering
        sys.exit("[ERROR] No features were retained after Spearman redundancy removal. Please check the threshold setting.")  # Exit if none remain

    selected_features, rel_df, selection_df = mrmr_select(X_spearman, y, top_k=TOP_K, criterion=MRMR_CRITERION)  # Step 3: perform mRMR feature selection
    if len(selected_features) == 0:  # Check whether mRMR selected any features
        sys.exit("[ERROR] mRMR failed to select any features. Please check the input data.")  # Exit if selection fails

    X_final = X_spearman[selected_features].copy()  # Extract the final selected features from the Spearman-reduced matrix
    save_final_dataset(df, y, X_final)  # Save the final modeling dataset

    summary_df = pd.DataFrame({  # Create a summary DataFrame
        "step": ["Original", "After_MWU", "After_Spearman", "After_mRMR"],  # Names of each selection step
        "feature_count": [X.shape[1], X_mwu.shape[1], X_spearman.shape[1], X_final.shape[1]]  # Number of features after each step
    })  # Finish constructing the summary table
    summary_df.to_csv(OUTPUT_DIR / "11_feature_count_summary.csv", index=False, encoding=ENCODING)  # Save the summary table

    print("\n[OK] Full pipeline completed successfully.")  # Print completion message
    print("Generated output files:")  # Print output title
    print("01_MWU_p_values.csv                     -> p-values of all features from Mann-Whitney U test")  # Describe output file
    print("02_MWU_significant_features.csv         -> significant features retained after MWU")  # Describe output file
    print("03_Spearman_correlation_matrix.csv      -> Spearman correlation matrix")  # Describe output file
    print("04_Spearman_dropped_features.csv        -> records of dropped redundant features")  # Describe output file
    print("05_Features_after_Spearman.csv          -> feature values after Spearman redundancy removal")  # Describe output file
    print("06_mRMR_relevance.csv                   -> mutual information relevance of each feature to the label")  # Describe output file
    print("07_mRMR_selected_features.txt           -> names of features selected by mRMR")  # Describe output file
    print("08_mRMR_selection_process.csv           -> step-by-step mRMR selection process")  # Describe output file
    print("09_mRMR_selected_feature_values.csv     -> values of final mRMR-selected features")  # Describe output file
    print("10_Final_selected_dataset.csv           -> final modeling dataset with the first two columns and selected features")  # Describe output file
    print("11_feature_count_summary.csv            -> summary of feature counts at each selection stage")  # Describe output file

if __name__ == "__main__":  # Check whether the script is being run directly
    main()  # Execute the main workflow