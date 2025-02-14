from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
def auc_computation(reference_anomaly_index, anomalous_anomaly_index):
    """
    Compute the AUC between reference anomaly scores and anomalous anomaly scores.

    Parameters:
    - reference_anomaly_index (array-like): Anomaly index scores from the reference (normal) data.
    - anomalous_anomaly_index (array-like): Anomaly index scores from the anomalous data.

    Returns:
    - float: The computed AUC score.
    """
    # Convert inputs to numpy arrays
    reference_anomaly_index = np.asarray(reference_anomaly_index)
    anomalous_anomaly_index = np.asarray(anomalous_anomaly_index)

    # Create labels: 0 for reference (normal), 1 for anomalies
    labels = np.concatenate([
        np.zeros(reference_anomaly_index.shape[0]),
        np.ones(anomalous_anomaly_index.shape[0])
    ])

    # Concatenate scores
    scores = np.concatenate([reference_anomaly_index, anomalous_anomaly_index])

    # Compute AUC
    auc_score = roc_auc_score(labels, scores)
    return auc_score

def compute_TPR(data,thresh:float)->float:
    """
    Compute the true positive rate
    """
    return np.sum(data>thresh)/len(data)

def label_dataframe_event(date_index, events):
    """
    Given a datetime index and an events dictionary, return a Series of event labels.
    
    Parameters
    ----------
    date_index : pd.Index
        A one-dimensional pandas Index containing datetime values.
    events : dict
        A dictionary of events. Each key is an event name, and its value is a dict with keys
        'start' and 'end' (values parseable by pd.to_datetime).
    
    Returns
    -------
    pd.Series
        A Series of labels with the same index as date_index. Dates not falling in any event
        are labeled as "no_event".
    """
    # Ensure the input index is in datetime format.
    dates = pd.to_datetime(date_index)
    
    # Initialize labels with a default value.
    labels = pd.Series("no_event", index=dates)
    
    # Update labels for each event.
    for event_name, event_info in events.items():
        start = pd.to_datetime(event_info['start'])
        end = pd.to_datetime(event_info['end'])
        mask = (dates >= start) & (dates <= end)
        labels.loc[mask] = event_name
        
    return labels


