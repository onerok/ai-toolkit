"""
ComfyUI Workflow Template Utilities

Handles loading, parsing, and injecting values into ComfyUI workflow JSON templates
for remote sampling.
"""

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

from toolkit.print import print_acc


class WorkflowTemplate:
    """
    Manages a ComfyUI workflow template with placeholder injection.

    Placeholders in the template should use double curly braces: {{placeholder_name}}

    Example placeholders:
        - {{prompt}} - The positive prompt
        - {{negative_prompt}} - The negative prompt
        - {{width}}, {{height}} - Image dimensions
        - {{steps}} - Number of sampling steps
        - {{guidance_scale}}, {{cfg}} - CFG scale
        - {{seed}} - Random seed
        - {{network_multiplier}} - LoRA strength
        - {{lora_url}} - URL to download the LoRA file
        - {{lora_filename}} - Filename of the LoRA
    """

    # Standard placeholders that should be recognized
    STANDARD_PLACEHOLDERS = {
        'prompt', 'negative_prompt', 'width', 'height',
        'steps', 'guidance_scale', 'cfg', 'seed',
        'network_multiplier', 'lora_url', 'lora_filename',
        'num_frames', 'fps', 'sampler_name', 'scheduler'
    }

    def __init__(self, template_path: Optional[str] = None, template_data: Optional[dict] = None):
        """
        Initialize the workflow template.

        Args:
            template_path: Path to a JSON workflow file
            template_data: Pre-loaded workflow dict (alternative to path)
        """
        self._template: Optional[dict] = None
        self._path: Optional[str] = None

        if template_path:
            self.load(template_path)
        elif template_data:
            self._template = template_data

    def load(self, template_path: str) -> bool:
        """
        Load a workflow template from a JSON file.

        Args:
            template_path: Path to the workflow JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                self._template = json.load(f)
            self._path = template_path
            return True
        except FileNotFoundError:
            print_acc(f"Workflow template not found: {template_path}")
            return False
        except json.JSONDecodeError as e:
            print_acc(f"Invalid JSON in workflow template: {e}")
            return False
        except Exception as e:
            print_acc(f"Error loading workflow template: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if a template is loaded."""
        return self._template is not None

    def get_placeholders(self) -> List[str]:
        """
        Find all placeholders in the template.

        Returns:
            List of placeholder names (without braces)
        """
        if not self._template:
            return []

        placeholders = set()
        pattern = re.compile(r'\{\{(\w+)\}\}')

        def find_in_value(value):
            if isinstance(value, str):
                matches = pattern.findall(value)
                placeholders.update(matches)
            elif isinstance(value, dict):
                for v in value.values():
                    find_in_value(v)
            elif isinstance(value, list):
                for item in value:
                    find_in_value(item)

        find_in_value(self._template)
        return sorted(list(placeholders))

    def validate(self) -> List[str]:
        """
        Validate the template and return any warnings.

        Returns:
            List of warning messages
        """
        warnings = []

        if not self._template:
            warnings.append("No template loaded")
            return warnings

        placeholders = set(self.get_placeholders())

        # Check for essential placeholders
        essential = {'prompt', 'lora_url', 'lora_filename'}
        missing_essential = essential - placeholders
        if missing_essential:
            warnings.append(f"Missing essential placeholders: {missing_essential}")

        # Check for unknown placeholders
        unknown = placeholders - self.STANDARD_PLACEHOLDERS
        if unknown:
            warnings.append(f"Unknown placeholders (may be intentional): {unknown}")

        # Check for AITKLoraDownloader node
        has_lora_downloader = False
        for node_id, node_data in self._template.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', '')
                if 'LoraDownloader' in class_type or 'lora' in class_type.lower():
                    has_lora_downloader = True
                    break

        if not has_lora_downloader and ('lora_url' in placeholders or 'lora_filename' in placeholders):
            warnings.append("Template uses LoRA placeholders but no LoRA loader node found")

        return warnings

    def build(self, **values) -> Optional[dict]:
        """
        Build a workflow by injecting values into the template.

        Args:
            **values: Placeholder values to inject

        Returns:
            The completed workflow dict, or None if template not loaded
        """
        if not self._template:
            return None

        # Deep copy the template
        workflow = copy.deepcopy(self._template)

        # Build replacement dict with {{}} syntax
        replacements = {f"{{{{{k}}}}}": v for k, v in values.items()}

        # Also support cfg as alias for guidance_scale
        if 'guidance_scale' in values and '{{cfg}}' not in replacements:
            replacements['{{cfg}}'] = values['guidance_scale']

        # Recursively replace placeholders
        workflow = self._replace_recursive(workflow, replacements)

        return workflow

    def _replace_recursive(self, value: Any, replacements: Dict[str, Any]) -> Any:
        """
        Recursively replace placeholders in a value.

        Args:
            value: The value to process
            replacements: Dict of placeholder -> replacement value

        Returns:
            The processed value
        """
        if isinstance(value, str):
            for placeholder, replacement in replacements.items():
                if placeholder in value:
                    # If entire value is the placeholder, replace with typed value
                    if value == placeholder:
                        return replacement
                    # Otherwise do string replacement
                    value = value.replace(placeholder, str(replacement))
            return value

        elif isinstance(value, dict):
            return {k: self._replace_recursive(v, replacements) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._replace_recursive(item, replacements) for item in value]

        return value

    def to_json(self, **values) -> Optional[str]:
        """
        Build workflow and return as JSON string.

        Args:
            **values: Placeholder values to inject

        Returns:
            JSON string of the workflow, or None if template not loaded
        """
        workflow = self.build(**values)
        if workflow:
            return json.dumps(workflow, indent=2)
        return None


def create_default_flux_workflow() -> dict:
    """
    Create a default workflow template for Flux models.

    This is a basic workflow that:
    1. Loads the checkpoint
    2. Downloads and loads the LoRA from AI Toolkit
    3. Encodes the prompt
    4. Samples with KSampler
    5. Decodes and saves the image

    Note: This requires the AITKLoraDownloader custom node to be installed in ComfyUI.
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
            "inputs": {
                "ckpt_name": "flux1-dev.safetensors"
            }
        },
        "2": {
            "class_type": "AITKLoraDownloader",
            "_meta": {"title": "Download LoRA from AI Toolkit"},
            "inputs": {
                "url": "{{lora_url}}",
                "filename": "{{lora_filename}}",
                "overwrite": True
            }
        },
        "3": {
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA"},
            "inputs": {
                "lora_name": ["2", 0],
                "strength_model": "{{network_multiplier}}",
                "strength_clip": "{{network_multiplier}}",
                "model": ["1", 0],
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
            "inputs": {
                "text": "{{prompt}}",
                "clip": ["3", 1]
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative)"},
            "inputs": {
                "text": "{{negative_prompt}}",
                "clip": ["3", 1]
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
            "inputs": {
                "width": "{{width}}",
                "height": "{{height}}",
                "batch_size": 1
            }
        },
        "7": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "seed": "{{seed}}",
                "steps": "{{steps}}",
                "cfg": "{{guidance_scale}}",
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["3", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
            "inputs": {
                "filename_prefix": "aitk_sample",
                "images": ["8", 0]
            }
        }
    }


def save_default_workflow(output_path: str, workflow_type: str = "flux") -> bool:
    """
    Save a default workflow template to a file.

    Args:
        output_path: Path to save the workflow JSON
        workflow_type: Type of workflow ("flux", etc.)

    Returns:
        True if saved successfully
    """
    if workflow_type == "flux":
        workflow = create_default_flux_workflow()
    else:
        print_acc(f"Unknown workflow type: {workflow_type}")
        return False

    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2)
        print_acc(f"Saved default {workflow_type} workflow to {output_path}")
        return True
    except Exception as e:
        print_acc(f"Error saving workflow: {e}")
        return False
