#!/usr/bin/env python3
"""
Lightweight caption service for on-demand image captioning.

Starts in a "cold" state with no model loaded. The model is loaded lazily
on the first caption request. Auto-shuts down after an idle timeout.

Supports dataset-specific config via .caption_config.yaml in the image's directory.

Usage:
    python scripts/caption_service.py [--port 5111] [--timeout 600]
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Global state
processor = None
model_state = "cold"  # cold, loading, ready
last_request_time = time.time()
idle_timeout = 600  # seconds (10 minutes default)
shutdown_event = threading.Event()


def get_processor():
    """Lazy-load the image processor and model."""
    global processor, model_state

    if processor is None:
        model_state = "loading"

        # Import here to avoid slow startup
        from extensions_built_in.dataset_tools.tools.qwen_vl_utils import QwenVLImageProcessor

        processor = QwenVLImageProcessor(device='cuda')
        processor.load_model()
        model_state = "ready"

    return processor


def get_default_prompt():
    """Get the default caption prompt."""
    from extensions_built_in.dataset_tools.tools.caption import default_long_prompt
    return default_long_prompt


def get_default_replacements():
    """Get the default caption replacements."""
    from extensions_built_in.dataset_tools.tools.caption import default_replacements
    return default_replacements


def load_dataset_config(image_path: str) -> dict:
    """
    Load dataset-specific caption config from .caption_config.yaml in the image's directory.

    Expected format:
        caption:
          prompt: "Your prompt here..."
          replacements:
            - ["find", "replace"]
            - ["another", ""]

    Returns a dict with 'prompt' and 'replacements' keys, using defaults for missing values.
    """
    config = {
        'prompt': get_default_prompt(),
        'replacements': get_default_replacements(),
    }

    # Look for .caption_config.yaml in the image's directory
    image_dir = Path(image_path).parent
    config_path = image_dir / '.caption_config.yaml'

    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config and 'caption' in yaml_config:
                caption_config = yaml_config['caption']
                if 'prompt' in caption_config:
                    config['prompt'] = caption_config['prompt']
                if 'replacements' in caption_config:
                    # Convert list of lists to list of tuples
                    config['replacements'] = [
                        tuple(r) if isinstance(r, list) else r
                        for r in caption_config['replacements']
                    ]
                print(f"Loaded dataset config from {config_path}")
        except Exception as e:
            print(f"Error loading dataset config from {config_path}: {e}")

    return config


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_state': model_state,
        'uptime': time.time() - app.config.get('start_time', time.time())
    })


@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint."""
    return jsonify({
        'model_state': model_state,
        'idle_timeout': idle_timeout,
        'seconds_since_last_request': time.time() - last_request_time
    })


@app.route('/caption', methods=['POST'])
def caption():
    """Generate a caption for an image."""
    global last_request_time
    last_request_time = time.time()

    data = request.json
    if not data or 'image_path' not in data:
        return jsonify({'error': 'image_path required'}), 400

    image_path = data['image_path']
    max_tokens = data.get('max_tokens', 512)

    # Load dataset-specific config (falls back to defaults)
    dataset_config = load_dataset_config(image_path)

    # Allow request to override config
    prompt = data.get('prompt', dataset_config['prompt'])
    replacements = data.get('replacements', dataset_config['replacements'])

    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Get processor (lazy loads model)
        proc = get_processor()

        # Generate caption
        caption_text = proc.generate_caption(
            image=image,
            prompt=prompt,
            replacements=replacements,
            max_new_tokens=max_tokens
        )

        return jsonify({
            'caption': caption_text,
            'model_state': model_state
        })

    except FileNotFoundError:
        return jsonify({'error': f'Image not found: {image_path}'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shutdown the service."""
    shutdown_event.set()
    return jsonify({'status': 'shutting down'})


def idle_monitor():
    """Monitor for idle timeout and shutdown if exceeded."""
    global last_request_time

    while not shutdown_event.is_set():
        time.sleep(10)  # Check every 10 seconds

        idle_time = time.time() - last_request_time
        if idle_time > idle_timeout:
            print(f"Idle timeout ({idle_timeout}s) exceeded. Shutting down.")
            shutdown_event.set()
            break

    # Give Flask a moment to respond to any pending requests
    time.sleep(1)
    import os
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description='Caption service for on-demand image captioning')
    parser.add_argument('--port', type=int, default=5111, help='Port to run on (default: 5111)')
    parser.add_argument('--timeout', type=int, default=600, help='Idle timeout in seconds (default: 600)')
    args = parser.parse_args()

    global idle_timeout
    idle_timeout = args.timeout

    app.config['start_time'] = time.time()

    # Start idle monitor thread
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()

    print(f"Caption service starting on port {args.port}")
    print(f"Idle timeout: {idle_timeout} seconds")
    print("Model will load on first caption request (cold start)")

    # Run Flask (use threaded=True for concurrent requests)
    app.run(host='127.0.0.1', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
