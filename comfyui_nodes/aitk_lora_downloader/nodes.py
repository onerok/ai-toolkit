"""
AI Toolkit LoRA Downloader Node

Downloads LoRA files from an AI Toolkit training server for use in ComfyUI workflows.
This enables remote sampling where the training GPU is freed while ComfyUI generates images.
"""

import os
import requests
import folder_paths


class AITKLoraDownloader:
    """
    Downloads a LoRA file from an AI Toolkit training server.

    The node fetches the LoRA from a URL and saves it to ComfyUI's loras folder,
    then outputs the filename for use with a LoraLoader node.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "http://localhost:8765/lora/remote_step_000001000.safetensors",
                    "multiline": False,
                    "placeholder": "http://training-server:8765/lora/filename.safetensors"
                }),
                "filename": ("STRING", {
                    "default": "aitk_remote_lora.safetensors",
                    "multiline": False,
                    "placeholder": "filename.safetensors"
                }),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 600,
                    "step": 10
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "download_lora"
    CATEGORY = "loaders"
    OUTPUT_NODE = False

    def download_lora(self, url: str, filename: str, overwrite: bool = True, timeout: int = 60):
        """
        Download a LoRA file from the given URL.

        Args:
            url: Full URL to the LoRA file on the AI Toolkit server
            filename: Filename to save as in the loras folder
            overwrite: Whether to overwrite if file exists
            timeout: Download timeout in seconds

        Returns:
            Tuple containing the lora_name for LoraLoader
        """
        # Get the loras folder path from ComfyUI
        loras_folder = folder_paths.get_folder_paths("loras")[0]

        # Ensure filename ends with .safetensors
        if not filename.endswith('.safetensors'):
            filename = filename + '.safetensors'

        # Full path to save the file
        lora_path = os.path.join(loras_folder, filename)

        # Check if file exists
        if os.path.exists(lora_path) and not overwrite:
            print(f"[AITK] LoRA already exists: {filename}")
            return (filename,)

        # Download the file
        try:
            print(f"[AITK] Downloading LoRA from: {url}")
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Get content length for progress
            total_size = int(response.headers.get('content-length', 0))

            # Write to file
            with open(lora_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

            print(f"[AITK] Downloaded LoRA: {filename} ({os.path.getsize(lora_path) / 1024 / 1024:.1f} MB)")
            return (filename,)

        except requests.exceptions.Timeout:
            raise RuntimeError(f"[AITK] Download timed out after {timeout}s: {url}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[AITK] Download failed: {e}")
        except IOError as e:
            raise RuntimeError(f"[AITK] Failed to save LoRA: {e}")


class AITKLoraDownloaderAdvanced:
    """
    Advanced version with more options for downloading LoRA files.

    Supports:
    - Custom headers (for authentication)
    - Subfolder organization
    - Retry logic
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "filename": ("STRING", {
                    "default": "aitk_remote_lora.safetensors",
                    "multiline": False
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "optional/subfolder"
                }),
                "overwrite": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {
                    "default": 60,
                    "min": 10,
                    "max": 600
                }),
                "retries": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10
                }),
                "auth_header": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Bearer your-token"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "download_lora"
    CATEGORY = "loaders"

    def download_lora(
        self,
        url: str,
        filename: str,
        subfolder: str = "",
        overwrite: bool = True,
        timeout: int = 60,
        retries: int = 3,
        auth_header: str = ""
    ):
        """Download LoRA with advanced options."""
        import time

        loras_folder = folder_paths.get_folder_paths("loras")[0]

        if not filename.endswith('.safetensors'):
            filename = filename + '.safetensors'

        # Handle subfolder
        if subfolder:
            save_folder = os.path.join(loras_folder, subfolder)
            os.makedirs(save_folder, exist_ok=True)
            lora_name = os.path.join(subfolder, filename)
        else:
            save_folder = loras_folder
            lora_name = filename

        lora_path = os.path.join(save_folder, filename)

        if os.path.exists(lora_path) and not overwrite:
            print(f"[AITK] LoRA already exists: {lora_name}")
            return (lora_name,)

        # Setup headers
        headers = {}
        if auth_header:
            headers['Authorization'] = auth_header

        # Download with retries
        last_error = None
        for attempt in range(retries):
            try:
                print(f"[AITK] Downloading LoRA (attempt {attempt + 1}/{retries}): {url}")
                response = requests.get(url, headers=headers, timeout=timeout, stream=True)
                response.raise_for_status()

                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"[AITK] Downloaded LoRA: {lora_name}")
                return (lora_name,)

            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"[AITK] Download failed after {retries} attempts: {last_error}")


# Additional node mappings for the advanced version
NODE_CLASS_MAPPINGS = {
    "AITKLoraDownloader": AITKLoraDownloader,
    "AITKLoraDownloaderAdvanced": AITKLoraDownloaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AITKLoraDownloader": "Download LoRA from AI Toolkit",
    "AITKLoraDownloaderAdvanced": "Download LoRA from AI Toolkit (Advanced)",
}
