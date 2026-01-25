"""
AI Toolkit LoRA Downloader Node for ComfyUI

This custom node allows ComfyUI to download LoRA files from an AI Toolkit
training session for remote sampling.

Installation:
1. Copy this folder to ComfyUI/custom_nodes/
2. Restart ComfyUI

Usage in workflows:
1. Add "AITKLoraDownloader" node
2. Connect to a LoraLoader node
3. The LoRA will be downloaded from the training server when the workflow runs
"""

from .nodes import AITKLoraDownloader

NODE_CLASS_MAPPINGS = {
    "AITKLoraDownloader": AITKLoraDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AITKLoraDownloader": "Download LoRA from AI Toolkit"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
