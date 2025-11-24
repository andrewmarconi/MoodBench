# Metric Development Plan for EmoBench

## Overview

This document outlines the implementation plan for adding additional metrics to the EmoBench benchmark process. The plan is structured in phases with clear priorities, dependencies, and implementation details.

## Current Metrics Status

The benchmark currently measures:
- **Classification**: accuracy, f1, precision, recall
- **Confusion Matrix**: specificity, sensitivity, FPR, FNR, true positives/negatives
- **AUC**: ROC-AUC, PR-AUC (binary classification)
- **Statistical**: Matthews correlation coefficient, Cohen's kappa
- **Performance**: latency (mean), throughput
- **Memory**: allocated, reserved, peak memory (CUDA/MPS/CPU)

## Additional Metrics to Implement

### Phase 1: Core Performance Metrics (High Priority)

#### 1. Enhanced Latency Metrics
**Location**: `src/evaluation/benchmark.py`

**Implementation Details**:
- Add Time to First Token (TTFT) measurement in `SpeedBenchmark.run_benchmark()`
- Collect full latency distributions instead of just means
- Calculate percentiles (P50, P95, P99) using numpy
- Update `BenchmarkRunner._evaluate_single_model()` to return percentile data

**Dependencies**: None
**Effort**: Medium (2-3 hours)

**Code Changes**:
```python
# In SpeedBenchmark.run_benchmark()
latencies = np.array(latencies)
latency_stats = {
    "mean_ms": np.mean(latencies) * 1000,
    "median_ms": np.median(latencies) * 1000,
    "p95_ms": np.percentile(latencies, 95) * 1000,
    "p99_ms": np.percentile(latencies, 99) * 1000,
    "ttft_ms": latencies[0] * 1000 if len(latencies) > 0 else None
}
```

#### 2. Advanced Classification Metrics
**Location**: `src/evaluation/metrics.py`

**Implementation Details**:
- Add `balanced_accuracy_score` to `MetricsCalculator.compute_metrics()`
- Implement macro/micro averaging for multi-class scenarios
- Add per-class metrics collection in `_compute_confusion_metrics()`
- Extend `get_classification_report()` to include balanced accuracy

**Dependencies**: sklearn.metrics
**Effort**: Low (1-2 hours)

**Code Changes**:
```python
# In MetricsCalculator.compute_metrics()
from sklearn.metrics import balanced_accuracy_score

metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)

# For multi-class
if self.num_classes > 2:
    metrics["macro_f1"] = f1_score(labels, predictions, average="macro")
    metrics["micro_f1"] = f1_score(labels, predictions, average="micro")
```

### Phase 2: Statistical Analysis Enhancement (Medium Priority)

#### 3. Statistical Significance Framework
**Location**: `src/comparison/statistical.py`

**Implementation Details**:
- Add effect size calculations to `paired_t_test()` and `wilcoxon_signed_rank_test()`
- Implement bootstrap confidence intervals in `bootstrap_ci()`
- Add multiple comparison correction methods (Bonferroni, Holm-Bonferroni)
- Create convenience methods for common statistical workflows

**Dependencies**: scipy.stats, statsmodels (optional)
**Effort**: Medium (3-4 hours)

**New Methods**:
```python
def effect_size_cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """Calculate Cohen's d effect size for two samples."""
    diff = np.mean(data1) - np.mean(data2)
    pooled_std = np.sqrt((np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2)
    return diff / pooled_std if pooled_std != 0 else 0.0
```

#### 4. Memory Profiling Extensions
**Location**: `src/evaluation/profiler.py`

**Implementation Details**:
- Add `get_model_memory()` integration to track parameter breakdown
- Implement temporal memory tracking across inference stages
- Add system memory monitoring (RAM, virtual memory)
- Create memory usage visualization helpers

**Dependencies**: psutil for system memory
**Effort**: Medium (2-3 hours)

**New Features**:
- Memory usage tracking over time during model loading, warmup, and inference
- System RAM and virtual memory monitoring
- Model parameter memory breakdown (trainable vs frozen)

### Phase 3: Advanced Features (Lower Priority)

#### 5. Hardware-Specific Metrics
**Location**: `src/evaluation/profiler.py`

**Implementation Details**:
- Add GPU utilization tracking (SM occupancy, memory bandwidth)
- Implement CPU affinity monitoring
- Add power consumption measurement (if hardware supports)
- Create hardware-specific metric collectors

**Dependencies**: pynvml (NVIDIA), psutil (CPU), platform-specific libraries
**Effort**: High (4-6 hours)

#### 6. Robustness Metrics
**Location**: New file `src/evaluation/robustness.py`

**Implementation Details**:
- Out-of-distribution detection using confidence thresholds
- Uncertainty quantification with prediction intervals
- Fairness metrics across demographic groups
- Adversarial robustness testing framework

**Dependencies**: Additional datasets, uncertainty libraries
**Effort**: High (6-8 hours)

### Phase 4: Integration and Testing

#### 7. Configuration Updates
**Location**: `config/evaluation.yaml`

**Implementation Details**:
- Add new metric enable/disable flags
- Configure statistical test parameters
- Set hardware monitoring options
- Define robustness evaluation settings

**New Configuration Sections**:
```yaml
advanced_metrics:
  enabled: true
  latency_percentiles: [50, 95, 99]
  statistical_tests:
    effect_sizes: true
    confidence_intervals: true
    multiple_comparison_correction: "bonferroni"
  memory_profiling:
    temporal_tracking: true
    system_memory: true
  robustness:
    ood_detection: false
    fairness_analysis: false
```

**Dependencies**: None
**Effort**: Low (1 hour)

#### 8. Testing and Validation
**Location**: `tests/test_evaluation.py`

**Implementation Details**:
- Unit tests for each new metric calculation
- Integration tests for benchmark pipeline
- Statistical test validation
- Memory profiling accuracy tests

**Test Structure**:
```python
def test_latency_percentiles():
    """Test latency percentile calculations."""
    latencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    percentiles = calculate_percentiles(latencies)
    assert percentiles["p95"] == 0.49  # 95th percentile

def test_balanced_accuracy():
    """Test balanced accuracy calculation."""
    predictions = [0, 1, 1, 0]
    labels = [0, 1, 0, 0]
    balanced_acc = balanced_accuracy_score(labels, predictions)
    assert balanced_acc == 0.75
```

**Dependencies**: pytest fixtures
**Effort**: Medium (3-4 hours)

#### 9. Results Integration
**Location**: `src/evaluation/results.py`, `src/visualization/`

**Implementation Details**:
- Update `ResultsAggregator` to handle new metric types
- Add visualization support for statistical metrics
- Create comparison plots for effect sizes and confidence intervals
- Extend dashboard to show robustness metrics

**Dependencies**: pandas, matplotlib/seaborn
**Effort**: Medium (2-3 hours)

#### 10. Documentation
**Location**: `docs/`, `README.md`

**Implementation Details**:
- Update metric definitions and usage examples
- Add statistical analysis guide
- Document hardware requirements for advanced metrics
- Create troubleshooting guide for metric collection

**Dependencies**: None
**Effort**: Low (2 hours)

## Implementation Guidelines

### Code Standards
- Follow existing patterns in `AGENTS.md` (type hints, logging, error handling)
- Use `src.utils.device.get_device()` for hardware detection
- Add comprehensive docstrings with Args/Returns sections
- Include example usage in module docstrings

### Testing Strategy
- Unit tests for metric calculations with known inputs/outputs
- Integration tests with actual model evaluation
- Statistical tests should use fixed seeds for reproducibility
- Memory tests should validate against system tools

### Backward Compatibility
- All new metrics should be optional (disabled by default)
- Existing metric calculations should remain unchanged
- Configuration should gracefully handle missing optional dependencies

### Performance Considerations
- Memory profiling should be optional due to overhead
- Statistical tests should be cached when possible
- Hardware metrics should only be collected when explicitly requested

## Timeline and Milestones

### Week 1: Core Metrics
- Implement enhanced latency metrics
- Add advanced classification metrics
- Update configuration
- Basic testing

### Week 2: Statistical Framework
- Implement statistical significance metrics
- Extend memory profiling
- Integration testing
- Results aggregation updates

### Week 3: Advanced Features (Optional)
- Hardware-specific metrics
- Robustness metrics
- Comprehensive testing
- Documentation updates

## Success Criteria

1. **Functionality**: All new metrics calculate correctly and integrate with existing pipeline
2. **Performance**: No significant overhead when metrics are disabled
3. **Compatibility**: Existing benchmarks continue to work unchanged
4. **Documentation**: Clear usage instructions and examples
5. **Testing**: Comprehensive test coverage for all new functionality

## Risk Mitigation

- **Incremental Implementation**: Each metric can be implemented and tested independently
- **Feature Flags**: All new metrics disabled by default to prevent breaking changes
- **Graceful Degradation**: System handles missing dependencies without crashing
- **Comprehensive Testing**: Extensive validation before production use