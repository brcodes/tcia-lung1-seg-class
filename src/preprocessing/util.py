import hashlib
import pandas as pd
import os

def file_sha256(obj):
    """
    Compute SHA256 hash for either a file path or a pandas DataFrame.

    Parameters
    ----------
    obj : str or pandas.DataFrame
        File path to CSV/manifest OR DataFrame object.

    Returns
    -------
    str
        SHA256 hex digest string.
    """
    
    # File path -> file hash case
    if isinstance(obj, str) and os.path.isfile(obj):
        h = hashlib.sha256()
        # File path case
        with open(obj, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # DataFrame case
    elif isinstance(obj, pd.DataFrame):
        # DataFrame case: hash values + index for reproducibility
        row_hashes = pd.util.hash_pandas_object(obj, index=True)
        return hashlib.sha256(row_hashes.values).hexdigest()

    else:
        raise TypeError("obj must be a file path or pandas DataFrame")

