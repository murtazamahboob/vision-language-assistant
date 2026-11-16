"""Vision-Language Assistant: ask questions about an image, grounded in pixels."""
from .analysis import Analysis, VisualObject, analyze, describe, nearest_color
from .benchmark import QUESTIONS, run_benchmark
from .image import Image, load_image, load_png, save_png
from .qa import Answer, ask
from .scenes import SCENES, make_scenes

__all__ = ["Analysis", "VisualObject", "analyze", "describe", "nearest_color",
           "QUESTIONS", "run_benchmark", "Image", "load_image", "load_png",
           "save_png", "Answer", "ask", "SCENES", "make_scenes"]
__version__ = "1.0.0"
