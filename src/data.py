from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(sample_size=50000, random_state=42):
    """
    Load and prepare the Forest Covertype dataset for model development.

    The function:
    1. Loads the full Covertype dataset.
    2. Creates a stratified development sample.
    3. Splits the sample into training, validation and test sets.
    4. Standardises the 10 continuous features using the training set only.
    5. Converts class labels from 1-7 to 0-6 for PyTorch.

    Returns the prepared train, validation and test datasets.
    """

    # ---------------------------------------------------------
    # 1. Load the full Forest Covertype dataset
    # ---------------------------------------------------------
    X, y = fetch_covtype(return_X_y=True)

    # ---------------------------------------------------------
    # 2. Create a smaller development sample
    #
    # Stratification is used so that the strong class imbalance
    # in the full dataset is preserved in the development sample.
    # ---------------------------------------------------------
    X_dev, _, y_dev, _ = train_test_split(
        X,
        y,
        train_size=sample_size,
        stratify=y,
        random_state=random_state
    )

    # ---------------------------------------------------------
    # 3. Split the development data
    #
    # First create:
    # 70% training data
    # 30% temporary data
    # ---------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_dev,
        y_dev,
        test_size=0.30,
        stratify=y_dev,
        random_state=random_state
    )

    # Divide the remaining 30% equally:
    # 15% validation
    # 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state
    )

    # ---------------------------------------------------------
    # 4. Standardise the continuous features
    #
    # The first 10 features are continuous measurements.
    # The remaining 44 features are already binary 0/1
    # wilderness-area and soil-type indicators, so these are
    # left unchanged.
    #
    # The scaler is fitted ONLY on the training data to avoid
    # leaking information from the validation or test sets.
    # ---------------------------------------------------------
    scaler = StandardScaler()

    # Copies are created so the original split arrays are not
    # modified directly.
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[:, :10] = scaler.fit_transform(X_train[:, :10])

    # Use the mean and standard deviation learned from X_train.
    X_val[:, :10] = scaler.transform(X_val[:, :10])
    X_test[:, :10] = scaler.transform(X_test[:, :10])

    # ---------------------------------------------------------
    # 5. Encode target labels
    #
    # Covertype labels are originally numbered 1-7.
    # PyTorch CrossEntropyLoss expects labels starting at 0,
    # so they are converted to 0-6.
    # ---------------------------------------------------------
    y_train = y_train - 1
    y_val = y_val - 1
    y_test = y_test - 1

    return X_train, X_val, X_test, y_train, y_val, y_test