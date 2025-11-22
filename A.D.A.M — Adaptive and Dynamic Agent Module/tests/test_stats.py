"""
Tests for VectLLM Statistics and Monitoring
"""

import pytest
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stats import (
    TrainingMetrics,
    StatsCollector,
    format_time,
    format_number,
)


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass"""

    def test_default_values(self):
        """Test default initialization values"""
        metrics = TrainingMetrics()

        assert metrics.total_cycles == 0
        assert metrics.total_tokens == 0
        assert metrics.current_loss == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.cycles_per_second == 0.0
        assert metrics.vocab_size == 0

    def test_loss_history_deque(self):
        """Test that loss_history is a deque with maxlen"""
        metrics = TrainingMetrics()

        assert hasattr(metrics.loss_history, 'maxlen')
        assert metrics.loss_history.maxlen == 1000


class TestStatsCollectorInit:
    """Tests for StatsCollector initialization"""

    def test_default_window_size(self):
        """Test default window size"""
        collector = StatsCollector()

        assert collector.window_size == 100

    def test_custom_window_size(self):
        """Test custom window size"""
        collector = StatsCollector(window_size=50)

        assert collector.window_size == 50

    def test_initial_state(self):
        """Test initial state"""
        collector = StatsCollector()

        assert collector.metrics is not None
        assert len(collector.loss_window) == 0
        assert len(collector.speed_window) == 0
        assert len(collector.checkpoint_metrics) == 0


class TestStatsCollectorUpdate:
    """Tests for update functionality"""

    def test_update_cycles(self):
        """Test updating cycles"""
        collector = StatsCollector()
        collector.update(cycles=100)

        assert collector.metrics.total_cycles == 100

    def test_update_tokens(self):
        """Test updating tokens"""
        collector = StatsCollector()
        collector.update(tokens=1000)

        assert collector.metrics.total_tokens == 1000

    def test_update_loss(self):
        """Test updating loss"""
        collector = StatsCollector()
        collector.update(loss=2.5)

        assert collector.metrics.current_loss == 2.5
        assert 2.5 in collector.loss_window

    def test_update_vocab_size(self):
        """Test updating vocabulary size"""
        collector = StatsCollector()
        collector.update(vocab_size=500)

        assert collector.metrics.vocab_size == 500

    def test_update_vocab_utilization(self):
        """Test updating vocabulary utilization"""
        collector = StatsCollector()
        collector.update(vocab_utilization=0.05)

        assert collector.metrics.vocab_utilization == 0.05

    def test_update_multiple_values(self):
        """Test updating multiple values at once"""
        collector = StatsCollector()
        collector.update(
            cycles=100,
            tokens=5000,
            loss=2.0,
            vocab_size=300
        )

        assert collector.metrics.total_cycles == 100
        assert collector.metrics.total_tokens == 5000
        assert collector.metrics.current_loss == 2.0
        assert collector.metrics.vocab_size == 300

    def test_loss_history_updated(self):
        """Test that loss history is updated"""
        collector = StatsCollector()

        collector.update(loss=3.0)
        collector.update(loss=2.5)
        collector.update(loss=2.0)

        assert len(collector.metrics.loss_history) == 3

    def test_tokens_per_second_calculation(self):
        """Test tokens per second calculation"""
        collector = StatsCollector()

        collector.update(tokens=0)
        time.sleep(0.1)
        collector.update(tokens=1000)

        # Should have calculated some speed
        assert collector.metrics.tokens_per_second > 0


class TestPerplexity:
    """Tests for perplexity calculation"""

    def test_get_perplexity(self):
        """Test perplexity calculation"""
        collector = StatsCollector()
        collector.update(loss=2.0)

        perplexity = collector.get_perplexity()

        expected = math.exp(2.0)
        assert abs(perplexity - expected) < 0.01

    def test_perplexity_zero_loss(self):
        """Test perplexity with zero loss"""
        collector = StatsCollector()

        perplexity = collector.get_perplexity()

        assert perplexity == float('inf')

    def test_perplexity_small_loss(self):
        """Test perplexity with small loss"""
        collector = StatsCollector()
        collector.update(loss=0.1)

        perplexity = collector.get_perplexity()

        assert perplexity > 1.0
        assert perplexity < 2.0


class TestAverageLoss:
    """Tests for average loss calculation"""

    def test_get_average_loss_empty(self):
        """Test average loss with no data"""
        collector = StatsCollector()

        avg = collector.get_average_loss()

        assert avg == 0.0

    def test_get_average_loss(self):
        """Test average loss calculation"""
        collector = StatsCollector()

        collector.update(loss=1.0)
        collector.update(loss=2.0)
        collector.update(loss=3.0)

        avg = collector.get_average_loss()

        assert avg == 2.0

    def test_get_average_loss_window(self):
        """Test average loss with specific window"""
        collector = StatsCollector()

        for i in range(10):
            collector.update(loss=float(i))

        # Average of last 3: 7, 8, 9
        avg = collector.get_average_loss(window=3)

        assert avg == 8.0


class TestElapsedTime:
    """Tests for elapsed time"""

    def test_get_elapsed_time(self):
        """Test elapsed time calculation"""
        collector = StatsCollector()

        time.sleep(0.1)
        elapsed = collector.get_elapsed_time()

        assert elapsed >= 0.1
        assert elapsed < 1.0


class TestGetSummary:
    """Tests for summary generation"""

    def test_get_summary_structure(self):
        """Test that summary has all required fields"""
        collector = StatsCollector()
        summary = collector.get_summary()

        required_fields = [
            'cycles', 'tokens', 'loss', 'loss_avg', 'perplexity',
            'tokens_per_sec', 'cycles_per_sec', 'vocab_size',
            'vocab_utilization', 'elapsed_time', 'elapsed_hours'
        ]

        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    def test_get_summary_values(self):
        """Test summary values"""
        collector = StatsCollector()
        collector.update(cycles=100, tokens=5000, loss=2.5, vocab_size=300)

        summary = collector.get_summary()

        assert summary['cycles'] == 100
        assert summary['tokens'] == 5000
        assert summary['loss'] == 2.5
        assert summary['vocab_size'] == 300


class TestPrintStats:
    """Tests for printing statistics"""

    def test_print_stats(self, capsys):
        """Test printing stats"""
        collector = StatsCollector()
        collector.update(cycles=1000, tokens=50000, loss=2.5, vocab_size=500)

        collector.print_stats()

        captured = capsys.readouterr()
        assert "1,000" in captured.out  # cycles
        assert "50,000" in captured.out  # tokens
        assert "2.5" in captured.out  # loss

    def test_print_stats_with_prefix(self, capsys):
        """Test printing stats with prefix"""
        collector = StatsCollector()
        collector.update(cycles=100)

        collector.print_stats(prefix="  ")

        captured = capsys.readouterr()
        assert "  Cycles:" in captured.out


class TestSaveCheckpointStats:
    """Tests for checkpoint statistics"""

    def test_save_checkpoint_stats(self):
        """Test saving checkpoint statistics"""
        collector = StatsCollector()
        collector.update(cycles=100, tokens=5000)

        collector.save_checkpoint_stats("test_checkpoint.ckpt")

        assert len(collector.checkpoint_metrics) == 1
        assert collector.checkpoint_metrics[0]['checkpoint_path'] == "test_checkpoint.ckpt"

    def test_checkpoint_metrics_structure(self):
        """Test checkpoint metrics structure"""
        collector = StatsCollector()
        collector.update(cycles=100)

        collector.save_checkpoint_stats("test.ckpt")

        metrics = collector.checkpoint_metrics[0]
        assert 'checkpoint_path' in metrics
        assert 'timestamp' in metrics
        assert 'metrics' in metrics


class TestGetLossTrend:
    """Tests for loss trend detection"""

    def test_trend_not_enough_data(self):
        """Test trend with not enough data"""
        collector = StatsCollector()

        for i in range(5):
            collector.update(loss=2.0)

        trend = collector.get_loss_trend()

        assert trend == "→"  # Stable/unknown

    def test_trend_improving(self):
        """Test detecting improving trend"""
        collector = StatsCollector()

        # Decreasing loss
        for i in range(20):
            collector.update(loss=5.0 - i * 0.2)

        trend = collector.get_loss_trend()

        assert trend == "↓"

    def test_trend_worsening(self):
        """Test detecting worsening trend"""
        collector = StatsCollector()

        # Increasing loss
        for i in range(20):
            collector.update(loss=1.0 + i * 0.2)

        trend = collector.get_loss_trend()

        assert trend == "↑"

    def test_trend_stable(self):
        """Test detecting stable trend"""
        collector = StatsCollector()

        # Stable loss
        for i in range(20):
            collector.update(loss=2.0)

        trend = collector.get_loss_trend()

        assert trend == "→"


class TestFormatTime:
    """Tests for format_time utility"""

    def test_format_seconds(self):
        """Test formatting seconds"""
        result = format_time(30.5)

        assert result == "30.5s"

    def test_format_minutes(self):
        """Test formatting minutes"""
        result = format_time(125)  # 2m 5s

        assert "2m" in result
        assert "5s" in result

    def test_format_hours(self):
        """Test formatting hours"""
        result = format_time(3700)  # 1h 1m

        assert "1h" in result
        assert "1m" in result

    def test_format_zero(self):
        """Test formatting zero"""
        result = format_time(0)

        assert result == "0.0s"


class TestFormatNumber:
    """Tests for format_number utility"""

    def test_format_small_number(self):
        """Test formatting small numbers"""
        assert format_number(500) == "500"

    def test_format_thousands(self):
        """Test formatting thousands"""
        result = format_number(1500)

        assert result == "1.5K"

    def test_format_millions(self):
        """Test formatting millions"""
        result = format_number(1500000)

        assert result == "1.5M"

    def test_format_billions(self):
        """Test formatting billions"""
        result = format_number(1500000000)

        assert result == "1.5B"


class TestStatsCollectorEdgeCases:
    """Tests for edge cases"""

    def test_window_size_limit(self):
        """Test that window respects size limit"""
        collector = StatsCollector(window_size=5)

        for i in range(10):
            collector.update(loss=float(i))

        assert len(collector.loss_window) == 5

    def test_rapid_updates(self):
        """Test rapid consecutive updates"""
        collector = StatsCollector()

        for i in range(100):
            collector.update(
                cycles=i,
                tokens=i * 100,
                loss=5.0 - i * 0.01
            )

        assert collector.metrics.total_cycles == 99
        assert collector.metrics.total_tokens == 9900

    def test_multiple_checkpoint_saves(self):
        """Test saving multiple checkpoint stats"""
        collector = StatsCollector()

        for i in range(5):
            collector.update(cycles=i * 100)
            collector.save_checkpoint_stats(f"checkpoint_{i}.ckpt")

        assert len(collector.checkpoint_metrics) == 5
