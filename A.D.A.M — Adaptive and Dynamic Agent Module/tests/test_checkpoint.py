"""
Tests for VectLLM Checkpoint Management
"""

import pytest
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.checkpoint import CheckpointManager, CheckpointInfo


class TestCheckpointInfo:
    """Tests for CheckpointInfo dataclass"""

    def test_create_checkpoint_info(self):
        """Test creating CheckpointInfo"""
        info = CheckpointInfo(
            version=3,
            magic="VECTLLM3",
            char_vocab_size=256,
            current_word_vocab_size=100,
            max_word_vocab_size=100000,
            embed_dim=768,
            num_layers=6,
            num_heads=12,
            num_clusters=256,
            total_cycles=5000,
            total_tokens=25000,
            timestamp=1234567890,
            learning_rate=0.0001,
            momentum=0.9,
            current_loss=2.5
        )

        assert info.version == 3
        assert info.magic == "VECTLLM3"
        assert info.total_cycles == 5000
        assert info.current_loss == 2.5


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization"""

    def test_default_directory(self):
        """Test default checkpoint directory"""
        manager = CheckpointManager()

        assert manager.checkpoint_dir == Path("./checkpoints")

    def test_custom_directory(self, temp_dir):
        """Test custom checkpoint directory"""
        custom_dir = temp_dir / "custom_checkpoints"
        manager = CheckpointManager(custom_dir)

        assert manager.checkpoint_dir == custom_dir
        assert custom_dir.exists()

    def test_creates_directory(self, temp_dir):
        """Test that directory is created if it doesn't exist"""
        new_dir = temp_dir / "new_checkpoints"
        assert not new_dir.exists()

        manager = CheckpointManager(new_dir)

        assert new_dir.exists()


class TestCheckpointPaths:
    """Tests for path generation"""

    def test_get_checkpoint_path_with_extension(self, temp_dir):
        """Test getting checkpoint path with .ckpt extension"""
        manager = CheckpointManager(temp_dir)
        path = manager.get_checkpoint_path("test.ckpt")

        assert path == temp_dir / "test.ckpt"

    def test_get_checkpoint_path_without_extension(self, temp_dir):
        """Test that .ckpt is added if missing"""
        manager = CheckpointManager(temp_dir)
        path = manager.get_checkpoint_path("test")

        assert path == temp_dir / "test.ckpt"

    def test_get_vocab_path(self, temp_dir):
        """Test getting vocabulary path"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        vocab_path = manager.get_vocab_path(ckpt_path)

        assert vocab_path == temp_dir / "test.vocab"

    def test_get_metadata_path(self, temp_dir):
        """Test getting metadata path"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        metadata_path = manager.get_metadata_path(ckpt_path)

        assert metadata_path == temp_dir / "test.json"


class TestSaveLoadMetadata:
    """Tests for saving and loading metadata"""

    def test_save_metadata(self, temp_dir, checkpoint_info):
        """Test saving metadata"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        manager.save_metadata(ckpt_path, checkpoint_info)

        metadata_path = manager.get_metadata_path(ckpt_path)
        assert metadata_path.exists()

    def test_load_metadata(self, temp_dir, checkpoint_info):
        """Test loading metadata"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        manager.save_metadata(ckpt_path, checkpoint_info)
        loaded = manager.load_metadata(ckpt_path)

        assert loaded is not None
        assert loaded["version"] == 3
        assert loaded["total_cycles"] == 5000
        assert loaded["current_loss"] == 2.5

    def test_load_metadata_nonexistent(self, temp_dir):
        """Test loading metadata for nonexistent checkpoint"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "nonexistent.ckpt"

        loaded = manager.load_metadata(ckpt_path)

        assert loaded is None

    def test_save_metadata_with_extra(self, temp_dir, checkpoint_info):
        """Test saving metadata with extra data"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        extra = {"note": "test checkpoint", "custom_field": 123}
        manager.save_metadata(ckpt_path, checkpoint_info, extra)

        loaded = manager.load_metadata(ckpt_path)

        assert "extra" in loaded
        assert loaded["extra"]["note"] == "test checkpoint"
        assert loaded["extra"]["custom_field"] == 123

    def test_metadata_is_valid_json(self, temp_dir, checkpoint_info):
        """Test that saved metadata is valid JSON"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        manager.save_metadata(ckpt_path, checkpoint_info)

        metadata_path = manager.get_metadata_path(ckpt_path)
        with open(metadata_path) as f:
            data = json.load(f)

        assert isinstance(data, dict)


class TestListCheckpoints:
    """Tests for listing checkpoints"""

    def test_list_empty_directory(self, temp_dir):
        """Test listing empty checkpoint directory"""
        manager = CheckpointManager(temp_dir)
        checkpoints = manager.list_checkpoints()

        assert checkpoints == []

    def test_list_checkpoints(self, temp_dir, checkpoint_info):
        """Test listing checkpoints"""
        manager = CheckpointManager(temp_dir)

        # Create some checkpoints
        for name in ["ckpt1", "ckpt2", "ckpt3"]:
            ckpt_path = temp_dir / f"{name}.ckpt"
            ckpt_path.touch()
            manager.save_metadata(ckpt_path, checkpoint_info)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3

    def test_list_checkpoints_with_metadata(self, temp_dir, checkpoint_info):
        """Test that list_checkpoints includes metadata"""
        manager = CheckpointManager(temp_dir)

        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()
        manager.save_metadata(ckpt_path, checkpoint_info)

        checkpoints = manager.list_checkpoints()

        path, metadata = checkpoints[0]
        assert path == ckpt_path
        assert metadata is not None
        assert metadata["version"] == 3

    def test_list_checkpoints_sorted(self, temp_dir, checkpoint_info):
        """Test that checkpoints are sorted"""
        manager = CheckpointManager(temp_dir)

        # Create checkpoints with names that sort alphabetically
        for name in ["c", "a", "b"]:
            ckpt_path = temp_dir / f"{name}.ckpt"
            ckpt_path.touch()

        checkpoints = manager.list_checkpoints()
        names = [p.stem for p, _ in checkpoints]

        assert names == ["a", "b", "c"]


class TestGetLatestCheckpoint:
    """Tests for getting latest checkpoint"""

    def test_get_latest_empty_directory(self, temp_dir):
        """Test getting latest from empty directory"""
        manager = CheckpointManager(temp_dir)
        latest = manager.get_latest_checkpoint()

        assert latest is None

    def test_get_latest_checkpoint(self, temp_dir):
        """Test getting latest checkpoint by modification time"""
        manager = CheckpointManager(temp_dir)

        # Create checkpoints with different times
        old = temp_dir / "old.ckpt"
        old.touch()

        time.sleep(0.1)  # Ensure different mtime

        new = temp_dir / "new.ckpt"
        new.touch()

        latest = manager.get_latest_checkpoint()

        assert latest == new


class TestDeleteCheckpoint:
    """Tests for deleting checkpoints"""

    def test_delete_checkpoint(self, temp_dir, checkpoint_info):
        """Test deleting checkpoint and associated files"""
        manager = CheckpointManager(temp_dir)

        # Create checkpoint with all files
        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()
        manager.save_metadata(ckpt_path, checkpoint_info)

        vocab_path = manager.get_vocab_path(ckpt_path)
        vocab_path.touch()

        # Delete
        manager.delete_checkpoint(ckpt_path)

        assert not ckpt_path.exists()
        assert not vocab_path.exists()
        assert not manager.get_metadata_path(ckpt_path).exists()

    def test_delete_checkpoint_only(self, temp_dir, checkpoint_info):
        """Test deleting only checkpoint file"""
        manager = CheckpointManager(temp_dir)

        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()
        manager.save_metadata(ckpt_path, checkpoint_info)

        manager.delete_checkpoint(ckpt_path, delete_vocab=False, delete_metadata=False)

        assert not ckpt_path.exists()
        assert manager.get_metadata_path(ckpt_path).exists()

    def test_delete_nonexistent_checkpoint(self, temp_dir):
        """Test deleting nonexistent checkpoint doesn't raise"""
        manager = CheckpointManager(temp_dir)
        ckpt_path = temp_dir / "nonexistent.ckpt"

        # Should not raise
        manager.delete_checkpoint(ckpt_path)


class TestCleanupOldCheckpoints:
    """Tests for cleanup functionality"""

    def test_cleanup_keeps_latest(self, temp_dir):
        """Test that cleanup keeps latest N checkpoints"""
        manager = CheckpointManager(temp_dir)

        # Create 5 checkpoints
        for i in range(5):
            ckpt_path = temp_dir / f"ckpt_{i}.ckpt"
            ckpt_path.touch()
            time.sleep(0.05)  # Ensure different mtimes

        # Keep only 2
        manager.cleanup_old_checkpoints(keep_latest=2)

        remaining = list(temp_dir.glob("*.ckpt"))
        assert len(remaining) == 2

    def test_cleanup_empty_directory(self, temp_dir):
        """Test cleanup on empty directory doesn't raise"""
        manager = CheckpointManager(temp_dir)

        # Should not raise
        manager.cleanup_old_checkpoints(keep_latest=5)

    def test_cleanup_fewer_than_keep(self, temp_dir):
        """Test cleanup when fewer checkpoints than keep_latest"""
        manager = CheckpointManager(temp_dir)

        # Create 2 checkpoints
        for i in range(2):
            ckpt_path = temp_dir / f"ckpt_{i}.ckpt"
            ckpt_path.touch()

        # Keep 5 (more than exist)
        manager.cleanup_old_checkpoints(keep_latest=5)

        remaining = list(temp_dir.glob("*.ckpt"))
        assert len(remaining) == 2


class TestPrintCheckpointInfo:
    """Tests for printing checkpoint info"""

    def test_print_checkpoint_info(self, temp_dir, checkpoint_info, capsys):
        """Test printing checkpoint info"""
        manager = CheckpointManager(temp_dir)

        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.write_bytes(b"dummy data")  # Add some content for size
        manager.save_metadata(ckpt_path, checkpoint_info)

        manager.print_checkpoint_info(ckpt_path)

        captured = capsys.readouterr()
        assert "test.ckpt" in captured.out
        assert "5,000" in captured.out  # cycles
        assert "2.5000" in captured.out  # loss

    def test_print_checkpoint_info_no_metadata(self, temp_dir, capsys):
        """Test printing info when no metadata exists"""
        manager = CheckpointManager(temp_dir)

        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        manager.print_checkpoint_info(ckpt_path)

        captured = capsys.readouterr()
        assert "No metadata" in captured.out


class TestCheckpointManagerEdgeCases:
    """Tests for edge cases"""

    def test_freq_file_deletion(self, temp_dir):
        """Test that .freq file is also deleted"""
        manager = CheckpointManager(temp_dir)

        ckpt_path = temp_dir / "test.ckpt"
        ckpt_path.touch()

        vocab_path = manager.get_vocab_path(ckpt_path)
        vocab_path.touch()

        freq_path = vocab_path.with_suffix('.freq')
        freq_path.touch()

        manager.delete_checkpoint(ckpt_path)

        assert not freq_path.exists()

    def test_concurrent_checkpoint_names(self, temp_dir, checkpoint_info):
        """Test handling checkpoints with similar names"""
        manager = CheckpointManager(temp_dir)

        # Create checkpoints with similar names
        for name in ["test", "test_1", "test_2"]:
            ckpt_path = temp_dir / f"{name}.ckpt"
            ckpt_path.touch()
            manager.save_metadata(ckpt_path, checkpoint_info)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 3
