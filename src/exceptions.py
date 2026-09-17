class ArtifactLoadError(Exception):
    """Raised when a model artifact fails to load from disk."""
    pass

class PredictionError(Exception):
    """Raised when prediction fails on otherwise valid input."""
    pass

class DataCleaningError(Exception):
    """Raised when the cleaning pipeline encounters unexpected input shape."""
    pass

