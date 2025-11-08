import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model


def build_batch_prompts(tokenizer, model, batch_size, conv_mode):
    """
    Build a batch of prompts with image tokens and conversation templates.
    Returns:
        questions: list[str]  (bare questions used as CLIP text input)
        prompts:   list[str]  (full conversation prompts)
    """
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    # simple dummy questions
    questions = [f"What is happening in image {i}?" for i in range(batch_size)]
    prompts = []

    for qs in questions:
        raw_q = qs
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
            else:
                qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)

    return questions, prompts


def tokenize_batch_prompts(prompts, tokenizer, device):
    """
    Use tokenizer_image_token for each prompt, then pad to a single tensor.
    Returns:
        input_ids: (B, L_max) LongTensor on `device`
    """
    id_lists = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        for p in prompts
    ]

    max_len = max(len(ids) for ids in id_lists)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    input_ids = torch.full(
        (len(prompts), max_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(id_lists):
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, : len(ids)] = ids_tensor

    return input_ids


def fabricate_random_images(batch_size, width=512, height=512):
    """
    Create a batch of random RGB PIL images.
    """
    images = []
    for _ in range(batch_size):
        arr = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HF id of the CDPruner LLaVA model (same as other scripts).",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Base model path for LoRA models (if applicable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of (image, text) pairs in the batch.",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="Conversation template key (e.g., llava_v1, llava_llama_2).",
    )
    parser.add_argument(
        "--visual-token-num",
        type=int,
        default=576,
        help="Number of visual tokens to keep after CDPruner.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # Load tokenizer, model, image processor as in existing eval scripts
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        visual_token_num=args.visual_token_num,
    )

    device = model.device if hasattr(model, "device") else "cuda"
    model.eval()

    # Build a batch of questions + prompts
    questions, prompts = build_batch_prompts(
        tokenizer, model, args.batch_size, args.conv_mode
    )

    # Tokenize prompts into a single batched tensor
    input_ids = tokenize_batch_prompts(prompts, tokenizer, device)

    # Fabricate a batch of random images
    images = fabricate_random_images(args.batch_size)
    image_sizes = [img.size for img in images]

    # IMPORTANT: process each image separately and pass a list of tensors
    image_tensors = []
    for img in images:
        # process_images expects a list of PIL images; returns a tensor
        t = process_images([img], image_processor, model.config)
        # move to device + dtype
        t = t.to(device=model.device, dtype=torch.float16)
        image_tensors.append(t)

    # Run batched generation with CDPruner
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_tensors,      # <-- list, not a single 4D tensor
            image_sizes=image_sizes,
            texts=questions,           # CLIP text tower uses the bare questions
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    # Handle both (output_ids, visual_token_num) and plain output_ids
    if isinstance(out, tuple):
        output_ids, visual_token_num = out
    else:
        output_ids = out
        visual_token_num = None

    # Decode per-sample
    print("\n=== Batched generation results ===")
    for i in range(output_ids.shape[0]):
        text = tokenizer.decode(
            output_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(f"\n[Sample {i}] Question: {questions[i]}")
        print(f"[Sample {i}] Output: {text}")
    if visual_token_num is not None:
        print(f"\n[CDPruner] visual_token_num reported by model: {visual_token_num}")


if __name__ == "__main__":
    main()
