from enum import Enum


class FinishReason(str, Enum):
    SUCCESS = "success"
    MAX_LENGTH = "max_length"
    STOP_SEQUENCE = "stop_sequence"
