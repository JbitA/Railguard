"""Cross-runtime model/data contracts.

Keep deployment-facing feature order explicit. A model checkpoint whose structured
features differ from this list is still valid for research, but it must not be exported
to the current native runtime without a corresponding contract/version change.
"""

TELEMETRY_SCHEMA_VERSION = 1
MODEL_ARCH_VERSION = 2
DEPLOYMENT_IMAGE_MODE = "monochrome_replicated_rgb"
DEPLOYMENT_SENSOR_COLUMNS = [
    "vibration_rms",
    "vibration_peak",
    "vibration_kurtosis",
    "crest_factor",
    "vision_motion",
    "vision_contrast",
    "speed_mps",
    "temperature_c",
    "humidity",
]
