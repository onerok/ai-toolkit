
from .caption import default_long_prompt, default_replacements, clean_caption

import torch
from PIL import Image


class QwenVLImageProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.model_path = "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated"

    def load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.is_loaded = True

    def generate_caption(
            self, image: Image,
            prompt: str = default_long_prompt,
            replacements=default_replacements,
            max_new_tokens=512
    ):
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output = output_text[0]
        return clean_caption(output, replacements=replacements)
