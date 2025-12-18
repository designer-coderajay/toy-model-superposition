"""
Tests for Toy Model of Superposition

Tests verify:
1. Model architecture and forward pass
2. Gradient computation correctness
3. Training convergence
4. Geometric properties of learned representations
5. Sparsity effects on superposition

Run with: python -m pytest test_superposition.py -v
"""

import numpy as np
import pytest
from toy_superposition import ToyModelSuperposition


class TestModelArchitecture:
    """Tests for basic model structure and forward pass."""
    
    def test_initialization(self):
        """Model initializes with correct shapes."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        
        assert model.W.shape == (2, 5), "Weight matrix should be (n_hidden, n_features)"
        assert model.importance.shape == (5,), "Importance should match n_features"
        assert model.n_features == 5
        assert model.n_hidden == 2
    
    def test_importance_decay(self):
        """Feature importance decays exponentially."""
        model = ToyModelSuperposition(n_features=5, importance_decay=0.5)
        
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(model.importance, expected)
    
    def test_forward_pass_shapes(self):
        """Forward pass produces correct output shapes."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        
        # Single sample
        x = np.random.randn(1, 5)
        x_hat = model.forward(x)
        assert x_hat.shape == (1, 5), "Output should match input shape"
        
        # Batch
        x_batch = np.random.randn(32, 5)
        x_hat_batch = model.forward(x_batch)
        assert x_hat_batch.shape == (32, 5), "Batch output should match batch input"
    
    def test_encode_decode_shapes(self):
        """Encode and decode have correct shapes."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = np.random.randn(16, 5)
        
        h = model.encode(x)
        assert h.shape == (2, 16), "Hidden should be (n_hidden, batch_size)"
        
        x_reconstructed = model.decode(h)
        assert x_reconstructed.shape == (16, 5), "Decoded should match input shape"


class TestLossComputation:
    """Tests for loss function correctness."""
    
    def test_zero_loss_perfect_reconstruction(self):
        """Loss is zero when reconstruction is perfect."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = np.random.randn(10, 5)
        
        # Perfect reconstruction
        loss = model.compute_loss(x, x)
        assert np.isclose(loss, 0), "Loss should be 0 for perfect reconstruction"
    
    def test_loss_positive(self):
        """Loss is always positive for imperfect reconstruction."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = np.random.randn(10, 5)
        x_hat = x + np.random.randn(*x.shape) * 0.1  # Add noise
        
        loss = model.compute_loss(x, x_hat)
        assert loss > 0, "Loss should be positive for imperfect reconstruction"
    
    def test_importance_weighting(self):
        """Loss weights errors by feature importance."""
        model = ToyModelSuperposition(n_features=2, n_hidden=1, importance_decay=0.0)
        # importance = [1.0, 0.0] - only first feature matters
        
        x = np.array([[1.0, 1.0]])
        
        # Error in first feature (important)
        x_hat_1 = np.array([[0.0, 1.0]])
        loss_1 = model.compute_loss(x, x_hat_1)
        
        # Error in second feature (not important)
        x_hat_2 = np.array([[1.0, 0.0]])
        loss_2 = model.compute_loss(x, x_hat_2)
        
        assert loss_1 > loss_2, "Error in important feature should cost more"


class TestGradients:
    """Tests for gradient computation."""
    
    def test_analytical_vs_numerical_gradient(self):
        """Analytical gradient matches numerical gradient."""
        model = ToyModelSuperposition(n_features=3, n_hidden=2, seed=42)
        x = model.generate_sparse_data(16, 0.5)
        
        analytical = model.compute_gradients_analytical(x)
        numerical = model.compute_gradients(x)
        
        np.testing.assert_array_almost_equal(
            analytical, numerical, decimal=4,
            err_msg="Analytical and numerical gradients should match"
        )
    
    def test_gradient_descent_reduces_loss(self):
        """Training step reduces loss."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        x = model.generate_sparse_data(64, 0.1)
        
        initial_loss = model.compute_loss(x, model.forward(x))
        
        # Take several training steps
        for _ in range(10):
            model.train_step(x, lr=0.1)
        
        final_loss = model.compute_loss(x, model.forward(x))
        
        assert final_loss < initial_loss, "Training should reduce loss"


class TestSparseDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_sparse_data_shape(self):
        """Generated data has correct shape."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = model.generate_sparse_data(100, 0.1)
        
        assert x.shape == (100, 5)
    
    def test_sparsity_level(self):
        """Generated data has approximately correct sparsity."""
        model = ToyModelSuperposition(n_features=10, n_hidden=2)
        
        for target_sparsity in [0.1, 0.3, 0.5]:
            x = model.generate_sparse_data(10000, target_sparsity)
            
            # Count non-zero entries
            actual_sparsity = np.mean(x > 0)
            
            # Should be within 5% of target
            assert abs(actual_sparsity - target_sparsity) < 0.05, \
                f"Sparsity should be ~{target_sparsity}, got {actual_sparsity}"
    
    def test_values_in_range(self):
        """Non-zero values are in [0, 1]."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = model.generate_sparse_data(1000, 0.5)
        
        assert np.all(x >= 0), "Values should be >= 0"
        assert np.all(x <= 1), "Values should be <= 1"


class TestTraining:
    """Tests for training behavior."""
    
    def test_training_converges(self):
        """Model converges to low loss with sparse features."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        model.train(sparsity=0.05, n_steps=2000, batch_size=128, 
                   lr=0.05, print_every=10000)
        
        # Final loss should be reasonably low
        assert model.loss_history[-1] < 0.1, "Model should converge to low loss"
    
    def test_loss_history_recorded(self):
        """Training records loss history."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        model.train(sparsity=0.1, n_steps=100, print_every=10000)
        
        assert len(model.loss_history) == 100, "Should record loss for each step"


class TestSuperpositionProperties:
    """Tests for superposition-specific properties."""
    
    def test_feature_vectors_extraction(self):
        """Can extract feature vectors from weights."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        vectors = model.get_feature_vectors()
        
        assert vectors.shape == (5, 2), "Should get 5 vectors in 2D"
    
    def test_more_features_than_dimensions_represented(self):
        """With sparsity, model represents more features than dimensions."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        model.train(sparsity=0.02, n_steps=5000, print_every=10000)
        
        vectors = model.get_feature_vectors()
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # With high sparsity, all features should have non-trivial representation
        non_trivial = np.sum(magnitudes > 0.1)
        
        assert non_trivial >= 3, \
            "With high sparsity, should represent more than 2 features in 2D"
    
    def test_dense_features_collapse_to_dimensions(self):
        """With dense features, model can only represent n_hidden features well."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        model.train(sparsity=1.0, n_steps=5000, print_every=10000)
        
        vectors = model.get_feature_vectors()
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # Sort by importance - top 2 should be represented
        sorted_mags = magnitudes[np.argsort(model.importance)[::-1]]
        
        # Top 2 features should have stronger representations
        assert sorted_mags[0] > sorted_mags[4], \
            "Most important features should be represented more strongly"
    
    def test_geometry_analysis(self):
        """Geometric analysis returns correct structure."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2, seed=42)
        model.train(sparsity=0.05, n_steps=3000, print_every=10000)
        
        analysis = model.analyze_geometry()
        
        assert 'magnitudes' in analysis
        assert 'pairwise_angles' in analysis
        assert 'angle_gaps' in analysis
        assert 'ideal_gap' in analysis
        
        # For 5 features, ideal gap is 72 degrees
        assert analysis['ideal_gap'] == 72.0


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_feature(self):
        """Works with single feature."""
        model = ToyModelSuperposition(n_features=1, n_hidden=1)
        x = model.generate_sparse_data(10, 0.5)
        x_hat = model.forward(x)
        
        assert x_hat.shape == x.shape
    
    def test_equal_features_and_dimensions(self):
        """Works when n_features == n_hidden (no superposition needed)."""
        model = ToyModelSuperposition(n_features=2, n_hidden=2, seed=42)
        model.train(sparsity=0.5, n_steps=1000, print_every=10000)
        
        # Should achieve very low loss since no bottleneck
        assert model.loss_history[-1] < 0.05
    
    def test_zero_sparsity_input(self):
        """Handles all-zero inputs gracefully."""
        model = ToyModelSuperposition(n_features=5, n_hidden=2)
        x = np.zeros((10, 5))
        
        x_hat = model.forward(x)
        loss = model.compute_loss(x, x_hat)
        
        assert np.allclose(x_hat, 0), "Zero input should give zero output"
        assert loss == 0, "Zero input should have zero loss"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
