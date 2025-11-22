"""
Tests for VectLLM Configuration System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import (
    ModelConfig,
    TrainingConfig,
    CheckpointConfig,
    RuntimeConfig,
    get_config_preset,
    set_config_from_preset,
    update_config,
    MODEL_CONFIG,
    TRAINING_CONFIG,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass"""

    def test_default_values(self):
        """Test that ModelConfig has correct default values"""
        config = ModelConfig()

        assert config.EMBED_DIM == 768
        assert config.NUM_HEADS == 12
        assert config.NUM_LAYERS == 6
        assert config.MAX_SEQ_LEN == 512
        assert config.CHAR_VOCAB_SIZE == 256
        assert config.MAX_WORD_VOCAB_SIZE == 100000
        assert config.WORD_CREATION_THRESHOLD == 5
        assert config.WORD_PRUNING_THRESHOLD == 2
        assert config.MAX_WORD_LENGTH == 20

    def test_custom_values(self):
        """Test creating ModelConfig with custom values"""
        config = ModelConfig(
            EMBED_DIM=512,
            NUM_HEADS=8,
            NUM_LAYERS=4,
        )

        assert config.EMBED_DIM == 512
        assert config.NUM_HEADS == 8
        assert config.NUM_LAYERS == 4

    def test_venn_semantic_defaults(self):
        """Test Venn semantic system defaults"""
        config = ModelConfig()

        assert config.VENN_CLUSTERS == 256
        assert config.INTERSECTION_THRESHOLD == 0.5
        assert config.CLUSTER_UPDATE_LR == 0.1
        assert config.EPISODIC_BUFFER_SIZE == 1024


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass"""

    def test_default_values(self):
        """Test that TrainingConfig has correct default values"""
        config = TrainingConfig()

        assert config.BASE_LR == 0.0001
        assert config.EMBEDDING_LR_SCALE == 0.1
        assert config.MOMENTUM == 0.9
        assert config.EXPLORATION_TEMPERATURE == 1.0

    def test_frequency_defaults(self):
        """Test update frequency defaults"""
        config = TrainingConfig()

        assert config.VENN_UPDATE_FREQUENCY == 100
        assert config.STATS_SYNC_FREQUENCY == 10
        assert config.VOCAB_PRUNING_FREQUENCY == 10000

    def test_sleep_timing_defaults(self):
        """Test sleep timing defaults"""
        config = TrainingConfig()

        assert config.SELF_LOOP_SLEEP_US == 5000
        assert config.EXTERNAL_INPUT_SLEEP_US == 100000


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass"""

    def test_default_values(self):
        """Test that CheckpointConfig has correct default values"""
        config = CheckpointConfig()

        assert config.CHECKPOINT_DIR == Path("checkpoints")
        assert config.AUTO_CHECKPOINT_INTERVAL == 100
        assert config.CHECKPOINT_VERSION == 3
        assert config.MAGIC_V3 == "VECTLLM3"
        assert config.MAGIC_V2 == "VECTLLM2"
        assert config.MAGIC_V1 == "VECTLLM"


class TestRuntimeConfig:
    """Tests for RuntimeConfig dataclass"""

    def test_default_values(self):
        """Test that RuntimeConfig has correct default values"""
        config = RuntimeConfig()

        assert config.DEVICE_ID == 0
        assert config.NVCC_ARCH == "auto"
        assert config.CACHE_DIR == Path.home() / ".cache" / "vectllm"

    def test_nvcc_flags_initialized(self):
        """Test that NVCC_FLAGS are initialized by __post_init__"""
        config = RuntimeConfig()

        assert config.NVCC_FLAGS is not None
        assert isinstance(config.NVCC_FLAGS, list)
        assert '-O3' in config.NVCC_FLAGS
        assert '--shared' in config.NVCC_FLAGS


class TestGetConfigPreset:
    """Tests for get_config_preset function"""

    def test_default_preset(self):
        """Test getting default preset"""
        preset = get_config_preset("default")

        assert "training" in preset
        assert "model" in preset
        assert isinstance(preset["training"], TrainingConfig)
        assert isinstance(preset["model"], ModelConfig)

    def test_fast_learning_preset(self):
        """Test fast_learning preset has higher LR"""
        preset = get_config_preset("fast_learning")

        assert preset["training"].BASE_LR == 0.001
        assert preset["training"].MOMENTUM == 0.7
        assert preset["training"].EXPLORATION_TEMPERATURE == 1.5
        assert preset["training"].VENN_UPDATE_FREQUENCY == 50

    def test_stable_preset(self):
        """Test stable preset has lower LR"""
        preset = get_config_preset("stable")

        assert preset["training"].BASE_LR == 0.00001
        assert preset["training"].MOMENTUM == 0.95
        assert preset["training"].EXPLORATION_TEMPERATURE == 0.7
        assert preset["training"].VENN_UPDATE_FREQUENCY == 200

    def test_inference_preset(self):
        """Test inference preset has zero LR"""
        preset = get_config_preset("inference")

        assert preset["training"].BASE_LR == 0.0
        assert preset["training"].MOMENTUM == 0.0
        assert preset["training"].EXPLORATION_TEMPERATURE == 0.5

    def test_research_preset(self):
        """Test research preset"""
        preset = get_config_preset("research")

        assert preset["training"].BASE_LR == 0.0005
        assert preset["training"].MOMENTUM == 0.8
        assert preset["training"].EXPLORATION_TEMPERATURE == 2.0
        assert preset["training"].VENN_UPDATE_FREQUENCY == 25

    def test_unknown_preset_returns_default(self):
        """Test that unknown preset returns default"""
        preset = get_config_preset("nonexistent_preset")
        default = get_config_preset("default")

        assert preset["training"].BASE_LR == default["training"].BASE_LR
        assert preset["training"].MOMENTUM == default["training"].MOMENTUM

    def test_all_presets_have_required_keys(self):
        """Test that all presets have required keys"""
        preset_names = ["default", "fast_learning", "stable", "inference", "research"]

        for name in preset_names:
            preset = get_config_preset(name)
            assert "training" in preset, f"Preset '{name}' missing 'training'"
            assert "model" in preset, f"Preset '{name}' missing 'model'"


class TestSetConfigFromPreset:
    """Tests for set_config_from_preset function"""

    def test_set_from_preset_changes_global_config(self):
        """Test that set_config_from_preset changes global configs"""
        # Store original
        import core.config as config_module
        original_lr = config_module.TRAINING_CONFIG.BASE_LR

        # Apply fast_learning preset
        set_config_from_preset("fast_learning")
        assert config_module.TRAINING_CONFIG.BASE_LR == 0.001

        # Restore default
        set_config_from_preset("default")
        assert config_module.TRAINING_CONFIG.BASE_LR == 0.0001


class TestUpdateConfig:
    """Tests for update_config function"""

    def test_update_valid_parameter(self):
        """Test updating a valid parameter"""
        import core.config as config_module

        original = config_module.TRAINING_CONFIG.BASE_LR
        update_config(BASE_LR=0.005)

        assert config_module.TRAINING_CONFIG.BASE_LR == 0.005

        # Restore
        update_config(BASE_LR=original)

    def test_update_multiple_parameters(self):
        """Test updating multiple parameters at once"""
        import core.config as config_module

        original_lr = config_module.TRAINING_CONFIG.BASE_LR
        original_momentum = config_module.TRAINING_CONFIG.MOMENTUM

        update_config(BASE_LR=0.01, MOMENTUM=0.5)

        assert config_module.TRAINING_CONFIG.BASE_LR == 0.01
        assert config_module.TRAINING_CONFIG.MOMENTUM == 0.5

        # Restore
        update_config(BASE_LR=original_lr, MOMENTUM=original_momentum)

    def test_update_invalid_parameter_raises_error(self):
        """Test that updating invalid parameter raises ValueError"""
        with pytest.raises(ValueError) as excinfo:
            update_config(INVALID_PARAM=123)

        assert "Unknown config parameter" in str(excinfo.value)
        assert "INVALID_PARAM" in str(excinfo.value)


class TestGlobalConfigs:
    """Tests for global configuration instances"""

    def test_global_configs_exist(self):
        """Test that global configs are initialized"""
        from core.config import MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG, RUNTIME_CONFIG

        assert MODEL_CONFIG is not None
        assert TRAINING_CONFIG is not None
        assert CHECKPOINT_CONFIG is not None
        assert RUNTIME_CONFIG is not None

    def test_global_configs_are_correct_types(self):
        """Test that global configs are correct types"""
        from core.config import MODEL_CONFIG, TRAINING_CONFIG, CHECKPOINT_CONFIG, RUNTIME_CONFIG

        assert isinstance(MODEL_CONFIG, ModelConfig)
        assert isinstance(TRAINING_CONFIG, TrainingConfig)
        assert isinstance(CHECKPOINT_CONFIG, CheckpointConfig)
        assert isinstance(RUNTIME_CONFIG, RuntimeConfig)
