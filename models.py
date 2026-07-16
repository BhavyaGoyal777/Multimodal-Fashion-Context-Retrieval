import json

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import open_clip
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSemanticSegmentation,
    AutoTokenizer,
    SegformerImageProcessor,
)

AXES = {
    # SegFormer labels:
    # 0=background, 4=upper clothes, 5=skirt, 6=pants, 7=dress.
    # Dresses are stored with the primary/upper garment representation.
    "upper": [4, 7],
    "lower": [5, 6],
    "background": [0],
}

MASK_FILL_VALUE = 127
DEFAULT_PARSER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class ClipImageEncoder(nn.Module):
    """Expose FashionCLIP image encoding as a standard forward method."""

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, pixel_values):
        return self.clip_model.encode_image(pixel_values)


class ModelLoader:
    def __init__(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.gpu_count = torch.cuda.device_count()

        if self.gpu_count:
            print(f"Detected {self.gpu_count} CUDA GPU(s):")
            for gpu_idx in range(self.gpu_count):
                print(f"  GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
        else:
            print("No CUDA GPU detected; using CPU.")

        self.seg_processor = None
        self.seg_model = None
        self.clip_model = None
        self.clip_image_encoder = None
        self.clip_preprocess = None
        self.tokenizer = None
        self.parser_model = None
        self.parser_tokenizer = None

    def load_segmentation_model(self):
        print("Loading SegFormer...")
        model_name = "mattmdjaga/segformer_b2_clothes"
        self.seg_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.seg_model = (
            AutoModelForSemanticSegmentation
            .from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

        if self.gpu_count > 1:
            self.seg_model = nn.DataParallel(
                self.seg_model,
                device_ids=list(range(self.gpu_count)),
            )
            print(f"SegFormer distributed across {self.gpu_count} GPUs")

        print("SegFormer loaded")

    def load_clip_model(self):
        print("Loading FashionCLIP...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionCLIP"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

        image_encoder = ClipImageEncoder(self.clip_model).to(self.device).eval()
        if self.gpu_count > 1:
            image_encoder = nn.DataParallel(
                image_encoder,
                device_ids=list(range(self.gpu_count)),
            )
            print(f"FashionCLIP image encoder distributed across {self.gpu_count} GPUs")

        self.clip_image_encoder = image_encoder
        print("FashionCLIP loaded")

    def load_query_parser(self, model_name=DEFAULT_PARSER_MODEL):
        """Load a small local text model for structured query parsing."""
        print(f"Loading query parser: {model_name}")
        self.parser_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parser_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.gpu_count else torch.float32,
            device_map="auto" if self.gpu_count else None,
        ).eval()
        print("Query parser loaded")

    def segment_image(self, image):
        """Segment one image. Kept as a convenience wrapper."""
        return self.segment_images([image])[0]

    def segment_images(self, images):
        """Segment a batch of PIL images and return one mask per image."""
        if not images:
            return []

        inputs = self.seg_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.seg_model(**inputs).logits

        # Interpolation on CPU is performed in float32 for compatibility.
        logits = logits.float().cpu()

        masks = []
        for image, image_logits in zip(images, logits):
            upsampled = nn.functional.interpolate(
                image_logits.unsqueeze(0),
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            masks.append(upsampled.argmax(dim=1)[0].numpy())
        return masks

    def extract_region_crop(self, image, masks, class_ids, padding_ratio=0.05):
        """Create a masked crop containing only the selected semantic region."""
        combined_mask = np.isin(masks, class_ids)
        if not combined_mask.any():
            return None

        rows = np.any(combined_mask, axis=1)
        cols = np.any(combined_mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        height, width = combined_mask.shape
        region_height = ymax - ymin + 1
        region_width = xmax - xmin + 1
        pad_y = int(region_height * padding_ratio)
        pad_x = int(region_width * padding_ratio)

        ymin = max(0, ymin - pad_y)
        ymax = min(height - 1, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(width - 1, xmax + pad_x)

        image_array = np.asarray(image)
        crop = image_array[ymin:ymax + 1, xmin:xmax + 1].copy()
        crop_mask = combined_mask[ymin:ymax + 1, xmin:xmax + 1]

        # Remove unrelated clothing/person/background pixels from the crop.
        crop[~crop_mask] = MASK_FILL_VALUE
        return Image.fromarray(crop)

    def encode_image(self, image):
        """Encode one image. Kept as a convenience wrapper."""
        return self.encode_images([image])[0]

    def encode_images(self, images):
        """Encode a batch of PIL images into normalized FashionCLIP vectors."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        img_tensor = torch.stack([
            self.clip_preprocess(image) for image in images
        ]).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            features = self.clip_image_encoder(img_tensor)

        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy()

    def encode_text(self, text):
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            features = self.clip_model.encode_text(text_tokens)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().flatten()

    def parse_query(self, query_text):
        """Parse a query into upper, lower, and background descriptions."""
        if self.parser_model is None or self.parser_tokenizer is None:
            raise RuntimeError("Call load_query_parser() before parse_query().")

        system_prompt = """You parse fashion retrieval queries into regional descriptions.

The complete original query is always matched against a global image embedding.
Only extract information useful for these three regional embeddings:

upper:
- shirts, t-shirts, sweaters, hoodies, jackets, blazers, coats, raincoats
- dresses
- attached details such as ties, collars and lapels

lower:
- pants, trousers, jeans, shorts and skirts

background:
- environments, places and settings

Rules:
- Return one valid JSON object and nothing else.
- Allowed keys: upper, lower, background.
- Preserve color-garment binding exactly.
- A dress belongs to upper.
- A tie belongs with its upper garment.
- Shoes, bags, watches, poses, actions, gender and identity are handled globally.
- Do not invent attributes.
- Omit axes that are not described.

Examples:
"bright yellow raincoat"
{"upper":"bright yellow raincoat"}

"blue jeans and white sneakers"
{"lower":"blue jeans"}

"black dress in a park"
{"upper":"black dress","background":"park"}

"blue shirt sitting on a park bench"
{"upper":"blue shirt","background":"park with bench"}

"red tie and white shirt in a formal setting"
{"upper":"white shirt with red tie","background":"formal setting"}

Return only JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query:\n{query_text}\n\nOutput:"},
        ]

        model_inputs = self.parser_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.parser_model.device)

        with torch.inference_mode():
            generated = self.parser_model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.parser_tokenizer.eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        response = self.parser_tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()
        response = response.replace("```json", "").replace("```", "").strip()

        object_start = response.find("{")
        object_end = response.rfind("}")
        if object_start == -1 or object_end == -1:
            print("Parser returned no JSON; using global retrieval only.")
            return {}

        try:
            parsed = json.loads(response[object_start:object_end + 1])
        except json.JSONDecodeError:
            print("Parser returned invalid JSON; using global retrieval only.")
            return {}

        if not isinstance(parsed, dict):
            return {}

        cleaned = {
            axis_name: value.strip()
            for axis_name, value in parsed.items()
            if (
                axis_name in AXES
                and isinstance(value, str)
                and value.strip()
            )
        }
        print(f"Parsed query: {cleaned}")
        return cleaned
