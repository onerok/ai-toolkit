"""
Remote ComfyUI Sampling Support

This module provides functionality to offload sample generation to remote ComfyUI
instances during training, freeing the training GPU for computation.

Samples do NOT affect training - they're purely for the user to monitor quality.
This means we can be aggressive with async/fire-and-forget approaches.
"""

import json
import os
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

import requests

# websocket-client is optional - only needed for websocket-based progress tracking
try:
    import websocket
except ImportError:
    websocket = None

from toolkit.print import print_acc


class RemoteSamplingMode(str, Enum):
    """Sampling mode for remote ComfyUI instances."""
    LOCAL = "local"               # Current behavior - sample on training GPU (default)
    ASYNC = "async"               # Continue training, fetch results when ready (for logging)
    FIRE_AND_FORGET = "fire_and_forget"  # Send to remote, don't track results
    SYNC = "sync"                 # Block training until remote samples complete
    FALLBACK = "fallback"         # Try remote, fall back to local if unavailable


@dataclass
class ComfyUIEndpoint:
    """Configuration for a single ComfyUI endpoint."""
    url: str
    name: Optional[str] = None

    def __post_init__(self):
        # Normalize URL - remove trailing slash
        self.url = self.url.rstrip('/')
        if self.name is None:
            self.name = self.url


@dataclass
class LoraServerConfig:
    """Configuration for the LoRA HTTP server."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8765
    external_url: Optional[str] = None  # URL that ComfyUI will use to reach this server

    def get_external_url(self) -> str:
        """Get the external URL for ComfyUI to access the LoRA server."""
        if self.external_url:
            return self.external_url
        return f"http://{self.host}:{self.port}"


@dataclass
class RemoteSamplingConfig:
    """Configuration for remote sampling via ComfyUI."""
    enabled: bool = False
    mode: RemoteSamplingMode = RemoteSamplingMode.LOCAL

    # ComfyUI endpoint(s) - single or pool
    endpoints: List[ComfyUIEndpoint] = field(default_factory=list)

    # LoRA server configuration
    lora_server: LoraServerConfig = field(default_factory=LoraServerConfig)

    # Workflow template (ComfyUI API format JSON file path)
    workflow_template: Optional[str] = None

    # Timeouts (in seconds)
    connect_timeout: int = 10
    generation_timeout: int = 600

    # Retry behavior
    max_retries: int = 2
    retry_delay: int = 5

    # LoRA format for ComfyUI (auto, comfy)
    lora_format: str = "auto"

    def __post_init__(self):
        # Convert mode string to enum if needed
        if isinstance(self.mode, str):
            self.mode = RemoteSamplingMode(self.mode)

        # Convert endpoint dicts to objects if needed
        if self.endpoints:
            self.endpoints = [
                ComfyUIEndpoint(**ep) if isinstance(ep, dict) else ep
                for ep in self.endpoints
            ]

        # Convert lora_server dict to object if needed
        if isinstance(self.lora_server, dict):
            self.lora_server = LoraServerConfig(**self.lora_server)


@dataclass
class PendingSample:
    """Tracks a pending async sample request."""
    step: int
    prompt_id: str
    client: 'ComfyUIClient'
    gen_configs: List[Any]
    submitted_at: float = field(default_factory=time.time)
    lora_filename: Optional[str] = None


@dataclass
class CompletedSample:
    """Result of a completed sample."""
    step: int
    images: List[Any]  # PIL Images or paths
    prompt_id: str
    elapsed_time: float


class ComfyUIClient:
    """Client for interacting with a single ComfyUI instance via HTTP/WebSocket."""

    def __init__(self, endpoint: ComfyUIEndpoint, connect_timeout: int = 10):
        self.endpoint = endpoint
        self.url = endpoint.url
        self.name = endpoint.name or endpoint.url
        self.connect_timeout = connect_timeout
        self.client_id = str(uuid.uuid4())
        self._ws: Optional[Any] = None  # websocket.WebSocket if websocket-client installed
        self._ws_lock = threading.Lock()

    def is_available(self) -> bool:
        """Check if the ComfyUI endpoint is reachable and has capacity."""
        try:
            response = requests.get(
                f"{self.url}/system_stats",
                timeout=self.connect_timeout
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_queue_status(self) -> Tuple[int, int]:
        """Get the queue status (pending, running) from ComfyUI."""
        try:
            response = requests.get(
                f"{self.url}/queue",
                timeout=self.connect_timeout
            )
            if response.status_code == 200:
                data = response.json()
                pending = len(data.get('queue_pending', []))
                running = len(data.get('queue_running', []))
                return pending, running
            return -1, -1
        except Exception:
            return -1, -1

    def queue_prompt(self, workflow: dict) -> Optional[str]:
        """
        Queue a prompt/workflow for execution.

        Args:
            workflow: The ComfyUI workflow in API format

        Returns:
            The prompt_id if successful, None otherwise
        """
        try:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            response = requests.post(
                f"{self.url}/prompt",
                json=payload,
                timeout=self.connect_timeout
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('prompt_id')
            else:
                print_acc(f"ComfyUI queue_prompt failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print_acc(f"ComfyUI queue_prompt error: {e}")
            return None

    def get_history(self, prompt_id: str) -> Optional[dict]:
        """
        Get the history/status of a queued prompt.

        Args:
            prompt_id: The prompt ID returned from queue_prompt

        Returns:
            The history data if available, None otherwise
        """
        try:
            response = requests.get(
                f"{self.url}/history/{prompt_id}",
                timeout=self.connect_timeout
            )
            if response.status_code == 200:
                data = response.json()
                return data.get(prompt_id)
            return None
        except Exception:
            return None

    def is_prompt_complete(self, prompt_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Check if a prompt has completed execution.

        Returns:
            Tuple of (is_complete, history_data)
        """
        history = self.get_history(prompt_id)
        if history is None:
            return False, None

        # Check if there are outputs (indicates completion)
        outputs = history.get('outputs', {})
        if outputs:
            return True, history

        return False, None

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Optional[bytes]:
        """
        Fetch an image from ComfyUI's output folder.

        Args:
            filename: The image filename
            subfolder: Optional subfolder within the output directory
            folder_type: Type of folder (output, input, temp)

        Returns:
            Image bytes if successful, None otherwise
        """
        try:
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }
            response = requests.get(
                f"{self.url}/view",
                params=params,
                timeout=self.connect_timeout * 2  # Images may take longer
            )
            if response.status_code == 200:
                return response.content
            return None
        except Exception:
            return None

    def wait_for_completion(self, prompt_id: str, timeout: int = 600, poll_interval: float = 1.0) -> Optional[dict]:
        """
        Wait for a prompt to complete, polling the history endpoint.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds

        Returns:
            The history data when complete, None if timeout or error
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            is_complete, history = self.is_prompt_complete(prompt_id)
            if is_complete:
                return history
            time.sleep(poll_interval)

        return None

    def extract_output_images(self, history: dict) -> List[Tuple[str, str]]:
        """
        Extract image filenames from completed history.

        Args:
            history: The history data from get_history

        Returns:
            List of (filename, subfolder) tuples for output images
        """
        images = []
        outputs = history.get('outputs', {})

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img_data in node_output['images']:
                    filename = img_data.get('filename', '')
                    subfolder = img_data.get('subfolder', '')
                    if filename:
                        images.append((filename, subfolder))

        return images

    def __repr__(self):
        return f"ComfyUIClient({self.name})"


class RemoteSamplingManager:
    """
    Manages remote sampling to ComfyUI instances.

    Handles:
    - Client pool management with round-robin selection
    - Async task tracking for ASYNC mode
    - LoRA file serving coordination
    - Workflow template injection
    """

    def __init__(
        self,
        config: RemoteSamplingConfig,
        save_root: str,
        network_ref: Optional[weakref.ref] = None,
        lora_server_ref: Optional[Any] = None
    ):
        self.config = config
        self.save_root = save_root
        self.network_ref = network_ref
        self.lora_server_ref = lora_server_ref

        # Initialize clients for all endpoints
        self.clients: List[ComfyUIClient] = [
            ComfyUIClient(ep, config.connect_timeout)
            for ep in config.endpoints
        ]

        # Round-robin index for client selection
        self._client_index = 0
        self._client_lock = threading.Lock()

        # Pending async samples
        self._pending_samples: List[PendingSample] = []
        self._pending_lock = threading.Lock()

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=len(self.clients) or 1)

        # Load workflow template if specified
        self._workflow_template: Optional[dict] = None
        if config.workflow_template:
            self._load_workflow_template(config.workflow_template)

    def _load_workflow_template(self, template_path: str):
        """Load the ComfyUI workflow template from a JSON file."""
        try:
            # Support relative paths from save_root or config directory
            if not os.path.isabs(template_path):
                # Try relative to save_root first
                full_path = os.path.join(self.save_root, template_path)
                if not os.path.exists(full_path):
                    # Try relative to current working directory
                    full_path = template_path
            else:
                full_path = template_path

            with open(full_path, 'r') as f:
                self._workflow_template = json.load(f)
            print_acc(f"Loaded workflow template from {full_path}")
        except Exception as e:
            print_acc(f"Warning: Failed to load workflow template: {e}")
            self._workflow_template = None

    def get_available_client(self) -> Optional[ComfyUIClient]:
        """
        Get an available ComfyUI client using round-robin selection.

        Returns:
            An available client, or None if no clients are available
        """
        if not self.clients:
            return None

        with self._client_lock:
            # Try each client starting from current index
            for _ in range(len(self.clients)):
                client = self.clients[self._client_index]
                self._client_index = (self._client_index + 1) % len(self.clients)

                if client.is_available():
                    return client

            return None

    def get_lora_url(self, lora_filename: str) -> str:
        """Get the URL for ComfyUI to download a LoRA file."""
        if self.lora_server_ref is not None:
            server = self.lora_server_ref
            if hasattr(server, 'get_file_url'):
                return server.get_file_url(lora_filename)

        # Fallback to constructing URL from config
        base_url = self.config.lora_server.get_external_url()
        return f"{base_url}/lora/{lora_filename}"

    def save_lora_for_remote(self, network: Any, step: int) -> str:
        """
        Save current LoRA weights to a file for remote access.

        Args:
            network: The LoRA network to save
            step: Current training step

        Returns:
            The filename of the saved LoRA
        """
        import torch

        # Create lora output directory
        lora_dir = os.path.join(self.save_root, 'remote_loras')
        os.makedirs(lora_dir, exist_ok=True)

        # Generate filename
        lora_filename = f"remote_step_{step:09d}.safetensors"
        lora_path = os.path.join(lora_dir, lora_filename)

        # Save the network weights
        try:
            metadata = OrderedDict({
                "step": str(step),
                "type": "remote_sample_lora"
            })
            network.save_weights(lora_path, dtype=torch.float16, metadata=metadata)
            print_acc(f"Saved LoRA for remote sampling: {lora_filename}")
        except Exception as e:
            print_acc(f"Error saving LoRA for remote: {e}")
            raise

        return lora_filename

    def build_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        network_multiplier: float,
        lora_filename: str,
        **extra_kwargs
    ) -> Optional[dict]:
        """
        Build a ComfyUI workflow from the template with injected values.

        Args:
            prompt: The positive prompt
            negative_prompt: The negative prompt
            width: Image width
            height: Image height
            steps: Number of sampling steps
            guidance_scale: CFG scale
            seed: Random seed
            network_multiplier: LoRA strength
            lora_filename: Name of the LoRA file
            **extra_kwargs: Additional parameters

        Returns:
            The workflow dict ready for ComfyUI, or None if no template
        """
        if self._workflow_template is None:
            print_acc("Warning: No workflow template configured")
            return None

        # Deep copy the template
        import copy
        workflow = copy.deepcopy(self._workflow_template)

        # Get LoRA URL
        lora_url = self.get_lora_url(lora_filename)

        # Values to inject
        replacements = {
            "{{prompt}}": prompt,
            "{{negative_prompt}}": negative_prompt,
            "{{width}}": width,
            "{{height}}": height,
            "{{steps}}": steps,
            "{{guidance_scale}}": guidance_scale,
            "{{seed}}": seed,
            "{{network_multiplier}}": network_multiplier,
            "{{lora_url}}": lora_url,
            "{{lora_filename}}": lora_filename,
            # Also support cfg alias
            "{{cfg}}": guidance_scale,
        }

        # Add extra kwargs as replacements
        for key, value in extra_kwargs.items():
            replacements[f"{{{{{key}}}}}"] = value

        # Recursively replace placeholders in the workflow
        def replace_in_value(value):
            if isinstance(value, str):
                for placeholder, replacement in replacements.items():
                    if placeholder in value:
                        # If the entire value is the placeholder, replace with the typed value
                        if value == placeholder:
                            return replacement
                        # Otherwise do string replacement
                        value = value.replace(placeholder, str(replacement))
                return value
            elif isinstance(value, dict):
                return {k: replace_in_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_in_value(item) for item in value]
            return value

        workflow = replace_in_value(workflow)
        return workflow

    def submit_sample(
        self,
        gen_config: Any,
        lora_filename: str,
        step: int,
        client: Optional[ComfyUIClient] = None
    ) -> Optional[str]:
        """
        Submit a single sample to ComfyUI.

        Args:
            gen_config: GenerateImageConfig with sample parameters
            lora_filename: Name of the LoRA file to use
            step: Current training step
            client: Specific client to use, or None for auto-selection

        Returns:
            The prompt_id if successful, None otherwise
        """
        if client is None:
            client = self.get_available_client()
            if client is None:
                print_acc("No available ComfyUI clients")
                return None

        # Build workflow
        workflow = self.build_workflow(
            prompt=gen_config.prompt,
            negative_prompt=gen_config.negative_prompt,
            width=gen_config.width,
            height=gen_config.height,
            steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            seed=gen_config.seed,
            network_multiplier=gen_config.network_multiplier,
            lora_filename=lora_filename
        )

        if workflow is None:
            return None

        # Submit to ComfyUI
        prompt_id = client.queue_prompt(workflow)
        return prompt_id

    def submit_samples_async(
        self,
        gen_configs: List[Any],
        lora_filename: str,
        step: int
    ) -> List[PendingSample]:
        """
        Submit multiple samples asynchronously.

        Args:
            gen_configs: List of GenerateImageConfig objects
            lora_filename: Name of the LoRA file
            step: Current training step

        Returns:
            List of PendingSample objects for tracking
        """
        pending = []

        for gen_config in gen_configs:
            client = self.get_available_client()
            if client is None:
                print_acc("Skipping sample - no available clients")
                continue

            prompt_id = self.submit_sample(gen_config, lora_filename, step, client)
            if prompt_id:
                sample = PendingSample(
                    step=step,
                    prompt_id=prompt_id,
                    client=client,
                    gen_configs=[gen_config],
                    lora_filename=lora_filename
                )
                pending.append(sample)

                with self._pending_lock:
                    self._pending_samples.append(sample)

        if pending:
            print_acc(f"Submitted {len(pending)} samples to remote ComfyUI (step {step})")

        return pending

    def submit_and_wait(
        self,
        gen_configs: List[Any],
        lora_filename: str,
        step: int,
        timeout: Optional[int] = None
    ) -> List[Any]:
        """
        Submit samples and wait for completion (SYNC mode).

        Args:
            gen_configs: List of GenerateImageConfig objects
            lora_filename: Name of the LoRA file
            step: Current training step
            timeout: Max time to wait (uses config default if None)

        Returns:
            List of PIL Images
        """
        from io import BytesIO

        from PIL import Image

        timeout = timeout or self.config.generation_timeout
        images = []

        for gen_config in gen_configs:
            client = self.get_available_client()
            if client is None:
                print_acc("Skipping sample - no available clients")
                continue

            prompt_id = self.submit_sample(gen_config, lora_filename, step, client)
            if not prompt_id:
                continue

            # Wait for completion
            history = client.wait_for_completion(prompt_id, timeout=timeout)
            if history is None:
                print_acc(f"Timeout waiting for sample (step {step})")
                continue

            # Extract and fetch images
            image_refs = client.extract_output_images(history)
            for filename, subfolder in image_refs:
                image_bytes = client.get_image(filename, subfolder)
                if image_bytes:
                    try:
                        img = Image.open(BytesIO(image_bytes))
                        images.append(img)
                    except Exception as e:
                        print_acc(f"Error loading image: {e}")

        return images

    def check_pending_samples(self) -> List[CompletedSample]:
        """
        Non-blocking check for completed async samples.

        Returns:
            List of CompletedSample objects for any completed samples
        """
        from io import BytesIO

        from PIL import Image

        completed = []
        to_remove = []

        with self._pending_lock:
            for sample in self._pending_samples:
                is_complete, history = sample.client.is_prompt_complete(sample.prompt_id)

                if is_complete and history:
                    elapsed = time.time() - sample.submitted_at
                    images = []

                    # Fetch images
                    image_refs = sample.client.extract_output_images(history)
                    for filename, subfolder in image_refs:
                        image_bytes = sample.client.get_image(filename, subfolder)
                        if image_bytes:
                            try:
                                img = Image.open(BytesIO(image_bytes))
                                images.append(img)
                            except Exception:
                                pass

                    completed.append(CompletedSample(
                        step=sample.step,
                        images=images,
                        prompt_id=sample.prompt_id,
                        elapsed_time=elapsed
                    ))
                    to_remove.append(sample)

                # Check for timeout
                elif time.time() - sample.submitted_at > self.config.generation_timeout:
                    print_acc(f"Remote sample timed out (step {sample.step})")
                    to_remove.append(sample)

            # Remove completed/timed out samples
            for sample in to_remove:
                self._pending_samples.remove(sample)

        return completed

    def get_pending_count(self) -> int:
        """Get the number of pending async samples."""
        with self._pending_lock:
            return len(self._pending_samples)

    def cleanup_old_loras(self, keep_recent: int = 5):
        """
        Clean up old LoRA files from the remote_loras directory.

        Args:
            keep_recent: Number of recent LoRA files to keep
        """
        lora_dir = os.path.join(self.save_root, 'remote_loras')
        if not os.path.exists(lora_dir):
            return

        try:
            files = sorted(
                [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')],
                key=lambda x: os.path.getmtime(os.path.join(lora_dir, x)),
                reverse=True
            )

            # Remove old files
            for f in files[keep_recent:]:
                try:
                    os.remove(os.path.join(lora_dir, f))
                except Exception:
                    pass
        except Exception:
            pass

    def shutdown(self):
        """Shutdown the manager and cleanup resources."""
        self._executor.shutdown(wait=False)

        # Clear pending samples
        with self._pending_lock:
            self._pending_samples.clear()
