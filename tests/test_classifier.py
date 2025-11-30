"""
Comprehensive unit tests for Global Average Pooling and Classifier Head.

Test Categories:
1. GAP Shape Tests - Verify pooling shapes
2. GAP Averaging Tests - Verify mean calculation
3. ClassifierHead Shape Tests - Verify output shapes
4. ClassifierHead Output Tests - Verify probabilities
5. Combined Classifier Tests - Verify end-to-end
6. Gradient Tests - Verify gradient flow
7. Edge Cases - Boundary conditions
8. Benchmark Tests - Performance measurements
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.blocks import GlobalAveragePooling, ClassifierHead, CombinedClassifier
from src.utils.testing import (
    check_output_shape,
    check_gradient_flow,
    check_no_nan_inf,
    count_parameters,
)


# ============================================================================
# Test Class 1: GAP Shape Tests
# ============================================================================

class TestGAPShape:
    """Test Global Average Pooling shapes."""
    
    def test_basic_shape(self):
        """Test basic shape transformation."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 196, 256)
        
        y = gap(x)
        
        assert y.shape == (2, 256)
    
    @pytest.mark.parametrize("seq_len", [16, 49, 100, 196, 400])
    def test_various_seq_lengths(self, seq_len):
        """Test with various sequence lengths."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, seq_len, 256)
        
        y = gap(x)
        
        assert y.shape == (2, 256)
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 49, embed_dim)
        
        y = gap(x)
        
        assert y.shape == (2, embed_dim)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        gap = GlobalAveragePooling()
        x = torch.randn(batch_size, 49, 256)
        
        y = gap(x)
        
        assert y.shape == (batch_size, 256)
    
    def test_single_token(self):
        """Test with single token sequence."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 1, 256)
        
        y = gap(x)
        
        assert y.shape == (2, 256)


# ============================================================================
# Test Class 2: GAP Averaging Tests
# ============================================================================

class TestGAPAveraging:
    """Test that GAP correctly computes mean."""
    
    def test_averaging_correct(self):
        """Verify GAP output equals manual mean."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 49, 256)
        
        y = gap(x)
        manual_mean = x.mean(dim=1)
        
        assert torch.allclose(y, manual_mean, atol=1e-6)
    
    def test_averaging_known_values(self):
        """Test with known values."""
        gap = GlobalAveragePooling()
        
        # Create tensor with known mean
        x = torch.ones(2, 4, 3)
        x[0, :, :] = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]).float()
        
        y = gap(x)
        
        # Expected: mean of each column
        expected_0 = torch.tensor([2.5, 3.5, 4.5])
        assert torch.allclose(y[0], expected_0, atol=1e-5)
    
    def test_single_token_passthrough(self):
        """Single token should pass through unchanged."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 1, 256)
        
        y = gap(x)
        
        assert torch.allclose(y, x.squeeze(1), atol=1e-6)
    
    def test_parameter_free(self):
        """Verify GAP has no parameters."""
        gap = GlobalAveragePooling()
        
        params = count_parameters(gap)
        
        assert params['total'] == 0
        assert params['trainable'] == 0


# ============================================================================
# Test Class 3: ClassifierHead Shape Tests
# ============================================================================

class TestClassifierHeadShape:
    """Test Classifier Head shapes."""
    
    def test_basic_shape(self):
        """Test basic shape transformation."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            y = classifier(x)
        
        assert y.shape == (2, 38)
    
    @pytest.mark.parametrize("num_classes", [10, 38, 100, 1000])
    def test_various_num_classes(self, num_classes):
        """Test with various number of classes."""
        classifier = ClassifierHead(embed_dim=256, num_classes=num_classes)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            y = classifier(x)
        
        assert y.shape == (2, num_classes)
    
    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512])
    def test_various_embed_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        classifier = ClassifierHead(embed_dim=embed_dim, num_classes=38)
        x = torch.randn(2, embed_dim)
        
        classifier.eval()
        with torch.no_grad():
            y = classifier(x)
        
        assert y.shape == (2, 38)
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_batch_size_variations(self, batch_size):
        """Test with various batch sizes."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(batch_size, 256)
        
        classifier.eval()
        with torch.no_grad():
            y = classifier(x)
        
        assert y.shape == (batch_size, 38)


# ============================================================================
# Test Class 4: ClassifierHead Output Tests
# ============================================================================

class TestClassifierHeadOutput:
    """Test Classifier Head output properties."""
    
    def test_probabilities_sum_to_one(self):
        """Verify probabilities sum to 1.0."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        prob_sums = probs.sum(dim=-1)
        
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-5)
    
    def test_probabilities_in_range(self):
        """Verify all probabilities are in [0, 1]."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_probabilities_non_negative(self):
        """Verify no negative probabilities."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(10, 256)  # Larger batch for better coverage
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert (probs >= 0).all()
    
    def test_logits_method(self):
        """Test get_logits returns raw scores."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier.get_logits(x)
            probs = classifier(x)
        
        # Logits should have same shape
        assert logits.shape == (2, 38)
        
        # Logits are raw scores (can be negative)
        # Softmax of logits should equal probs
        expected_probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs, expected_probs, atol=1e-5)
    
    def test_argmax_prediction(self):
        """Test that argmax gives valid class indices."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        predicted_classes = probs.argmax(dim=-1)
        
        assert predicted_classes.shape == (2,)
        assert (predicted_classes >= 0).all()
        assert (predicted_classes < 38).all()


# ============================================================================
# Test Class 5: Dropout Tests
# ============================================================================

class TestClassifierDropout:
    """Test dropout behavior in ClassifierHead."""
    
    def test_dropout_train_mode(self):
        """Test dropout is active in train mode."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.5)
        classifier.train()
        
        x = torch.randn(2, 256)
        
        outputs = [classifier(x).clone() for _ in range(5)]
        
        # Outputs should differ due to dropout
        all_same = all(torch.allclose(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Outputs should differ in train mode with dropout"
    
    def test_dropout_eval_mode(self):
        """Test dropout is inactive in eval mode."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.5)
        classifier.eval()
        
        x = torch.randn(2, 256)
        
        with torch.no_grad():
            y1 = classifier(x).clone()
            y2 = classifier(x).clone()
        
        assert torch.allclose(y1, y2), "Outputs should be identical in eval mode"
    
    def test_dropout_zero(self):
        """Test with dropout=0 (no dropout)."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.0)
        classifier.train()
        
        x = torch.randn(2, 256)
        
        y1 = classifier(x).clone()
        y2 = classifier(x).clone()
        
        assert torch.allclose(y1, y2), "With dropout=0, outputs should be identical"


# ============================================================================
# Test Class 6: Combined Classifier Tests
# ============================================================================

class TestCombinedClassifier:
    """Test Combined GAP + ClassifierHead."""
    
    def test_basic_shape(self):
        """Test basic shape transformation."""
        classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        x = torch.randn(2, 196, 256)
        
        classifier.eval()
        with torch.no_grad():
            y = classifier(x)
        
        assert y.shape == (2, 38)
    
    def test_probabilities_valid(self):
        """Verify output probabilities are valid."""
        classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        x = torch.randn(2, 196, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        # Sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
        
        # In [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_get_logits(self):
        """Test get_logits method."""
        classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        x = torch.randn(2, 196, 256)
        
        classifier.eval()
        with torch.no_grad():
            logits = classifier.get_logits(x)
        
        assert logits.shape == (2, 38)
    
    def test_matches_separate_modules(self):
        """Verify combined matches separate GAP + Head."""
        combined = CombinedClassifier(embed_dim=256, num_classes=38)
        gap = GlobalAveragePooling()
        
        # Copy weights to ensure same computation
        head = ClassifierHead(embed_dim=256, num_classes=38)
        head.fc.weight.data = combined.head.fc.weight.data.clone()
        head.fc.bias.data = combined.head.fc.bias.data.clone()
        
        x = torch.randn(2, 49, 256)
        
        combined.eval()
        head.eval()
        
        with torch.no_grad():
            y_combined = combined(x)
            y_separate = head(gap(x))
        
        assert torch.allclose(y_combined, y_separate, atol=1e-6)


# ============================================================================
# Test Class 7: Gradient Tests
# ============================================================================

class TestClassifierGradient:
    """Test gradient flow for classifier components."""
    
    def test_classifier_gradient_flow(self):
        """Verify ClassifierHead parameters receive gradients."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        grad_info = check_gradient_flow(classifier, x)
        
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_combined_gradient_flow(self):
        """Verify CombinedClassifier parameters receive gradients."""
        classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        x = torch.randn(2, 49, 256)
        
        grad_info = check_gradient_flow(classifier, x)
        
        for name, has_grad in grad_info.items():
            assert has_grad, f"Parameter {name} did not receive gradient"
    
    def test_gradient_with_loss(self):
        """Test gradient computation with cross-entropy loss."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        target = torch.randint(0, 38, (2,))
        
        # Use logits for CrossEntropyLoss
        logits = classifier.get_logits(x)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Check gradients exist
        assert classifier.fc.weight.grad is not None
        assert classifier.fc.weight.grad.abs().sum() > 0
    
    def test_gap_passthrough_gradient(self):
        """Verify gradients pass through GAP to input."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 49, 256, requires_grad=True)
        
        y = gap(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ============================================================================
# Test Class 8: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_class(self):
        """Test with single class (binary-ish)."""
        classifier = ClassifierHead(embed_dim=256, num_classes=1)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert probs.shape == (2, 1)
        # Single class softmax always outputs 1.0
        assert torch.allclose(probs, torch.ones(2, 1), atol=1e-5)
    
    def test_two_classes(self):
        """Test with two classes (binary)."""
        classifier = ClassifierHead(embed_dim=256, num_classes=2)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert probs.shape == (2, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    
    def test_many_classes(self):
        """Test with many classes."""
        classifier = ClassifierHead(embed_dim=256, num_classes=1000)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert probs.shape == (2, 1000)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)
    
    def test_small_embed_dim(self):
        """Test with small embedding dimension."""
        classifier = ClassifierHead(embed_dim=8, num_classes=38)
        x = torch.randn(2, 8)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        assert probs.shape == (2, 38)
    
    def test_no_nan_inf(self):
        """Verify no NaN or Inf in output."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        x = torch.randn(2, 256)
        
        classifier.eval()
        with torch.no_grad():
            probs = classifier(x)
        
        check_no_nan_inf(probs, "classifier output")
    
    def test_get_config(self):
        """Test configuration retrieval."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38, dropout=0.1)
        
        config = classifier.get_config()
        
        assert config['class'] == 'ClassifierHead'
        assert config['embed_dim'] == 256
        assert config['num_classes'] == 38
        assert config['dropout'] == 0.1


# ============================================================================
# Test Class 9: Parameter Count Tests
# ============================================================================

class TestParameterCount:
    """Test parameter counts."""
    
    def test_gap_has_zero_params(self):
        """GAP should have exactly 0 parameters."""
        gap = GlobalAveragePooling()
        
        params = count_parameters(gap)
        
        assert params['total'] == 0
    
    def test_classifier_param_count(self):
        """Verify classifier parameter count."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        
        params = count_parameters(classifier)
        
        # weight: 256 * 38 = 9728
        # bias: 38
        # total: 9766
        expected = 256 * 38 + 38
        assert params['total'] == expected
    
    def test_combined_param_count(self):
        """Verify combined classifier parameter count."""
        combined = CombinedClassifier(embed_dim=256, num_classes=38)
        
        params = count_parameters(combined)
        
        # Same as ClassifierHead (GAP has 0)
        expected = 256 * 38 + 38
        assert params['total'] == expected


# ============================================================================
# Test Class 10: Benchmark Tests
# ============================================================================

class TestBenchmark:
    """Performance measurements."""
    
    def test_gap_forward_time(self):
        """Measure GAP forward pass time."""
        gap = GlobalAveragePooling()
        x = torch.randn(2, 196, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = gap(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = gap(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nGAP forward time: {mean_time:.3f} ms")
        
        assert mean_time < 10, f"GAP too slow: {mean_time:.3f} ms"
    
    def test_classifier_forward_time(self):
        """Measure classifier forward pass time."""
        classifier = ClassifierHead(embed_dim=256, num_classes=38)
        classifier.eval()
        x = torch.randn(2, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = classifier(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = classifier(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nClassifier forward time: {mean_time:.3f} ms")
        
        assert mean_time < 10, f"Classifier too slow: {mean_time:.3f} ms"
    
    def test_combined_forward_time(self):
        """Measure combined classifier forward pass time."""
        classifier = CombinedClassifier(embed_dim=256, num_classes=38)
        classifier.eval()
        x = torch.randn(2, 196, 256)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = classifier(x)
        
        import time
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = classifier(x)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = sum(times) / len(times)
        print(f"\nCombined classifier forward time: {mean_time:.3f} ms")
        
        assert mean_time < 20, f"Combined too slow: {mean_time:.3f} ms"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])