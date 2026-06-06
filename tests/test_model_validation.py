import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestModelValidation:
    """Test suite for model validation including predictions and performance."""

    @pytest.fixture
    def sample_data(self):
        """Create a small sample dataset for testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=13,
            n_informative=10,
            n_redundant=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_model_prediction_type_and_shape(self, sample_data):
        """Test that model produces predictions of the correct type and shape."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model on small sample
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # Verify prediction type
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert predictions.dtype in [np.int32, np.int64, int], "Predictions should be integer type"
        
        # Verify prediction shape
        assert predictions.shape == (X_test.shape[0],), \
            f"Predictions shape {predictions.shape} doesn't match test set size {X_test.shape[0]}"
        
        # Verify probability shape
        assert probabilities.shape == (X_test.shape[0], 2), \
            f"Probabilities shape {probabilities.shape} should be (n_samples, n_classes)"

    def test_model_achieves_minimum_performance(self, sample_data):
        """Test that model achieves minimum performance threshold on test set."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Train model on small sample
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Set minimum performance threshold (50% for this test)
        min_threshold = 0.5
        
        assert accuracy >= min_threshold, \
            f"Model accuracy {accuracy:.3f} is below minimum threshold {min_threshold}"
        
        # Additional check: ensure accuracy is reasonable (not just lucky)
        assert accuracy <= 1.0, "Accuracy cannot exceed 1.0"
