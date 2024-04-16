from enum import Enum


class TemplateStyle(str, Enum):
    JINJA = "JINJA"
    PYTHON_STRING = "PYTHON_STRING"
    PYTHON_TEMPLATE = "PYTHON_TEMPLATE"
