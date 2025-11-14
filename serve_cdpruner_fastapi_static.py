#!/usr/bin/env python
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional
from uuid import uuid4

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import os
from llava.serve.profiler import run_profiler, set_visual_tokens
SCHEDULER_PROFILE = None  # filled at startup
# Fixed SLO for all requests (ms)
REQUEST_SLO_MS = float(os.getenv("CDPRUNER_REQUEST_SLO_MS", "300"))

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

class GenerateRequest(BaseModel):
    prompt: str
    image_path: Optional[str] = None


class GenerateResponse(BaseModel):
    id: str
    output: str

@dataclass
class QueueItem:
    id: str
    request: GenerateRequest
    future: asyncio.Future
    arrival_time: float


@dataclass
class QueueMetrics:
    queue_length: int
    oldest_wait: float
    avg_wait: float
    newest_wait: float
    now: float

app = FastAPI()

pending: Deque[QueueItem] = deque()
condition = asyncio.Condition()

tokenizer = None
model = None
image_processor = None
conv_mode = "llava_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_BATCH_SIZE = 8
vtn = 256

def build_prompts_and_questions(
    requests: List[QueueItem],
) -> (List[str], List[str]):
    """
    For each request, build:
      - question: bare question string (for CLIP text tower)
      - prompt: full conversation prompt with <image> token injected
    """
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    questions: List[str] = []
    prompts: List[str] = []

    for item in requests:
        raw_q = item.request.prompt
        qs = raw_q

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

        questions.append(raw_q)
        prompts.append(prompt)

    return questions, prompts


def tokenize_prompts(prompts: List[str]) -> torch.Tensor:
    id_lists = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        for p in prompts
    ]
    max_len = max(len(ids) for ids in id_lists)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

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


def load_and_process_images(requests: List[QueueItem]) -> (List[torch.Tensor], List[tuple]):
    """
    For MVP: assume each request has exactly one image_path pointing to a local file.
    Returns:
      - image_tensors: list of (1, 3, H, W) tensors on `device`
      - image_sizes: list of (width, height) tuples
    """
    from PIL import Image

    images = []
    image_sizes = []
    for item in requests:
        if item.request.image_path is None:
            raise ValueError(f"Request {item.id} missing image_path")
        img = Image.open(item.request.image_path).convert("RGB")
        images.append(img)
        image_sizes.append(img.size)

    image_tensors: List[torch.Tensor] = []
    for img in images:
        t = process_images([img], image_processor, model.config)
        t = t.to(device=device, dtype=torch.float16)
        image_tensors.append(t)

    return image_tensors, image_sizes


def batched_generate(requests: List[QueueItem]) -> List[str]:
    """
    Core CDPruner+LLaVA batched generate.
    Returns one decoded string per request, same order as `requests`.
    """
    questions, prompts = build_prompts_and_questions(requests)
    input_ids = tokenize_prompts(prompts)

    image_tensors, image_sizes = load_and_process_images(requests)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            texts=questions,
            do_sample=False,
            max_new_tokens=64,
        )

    if isinstance(out, tuple):
        output_ids = out[0]
    else:
        output_ids = out

    outputs: List[str] = []
    for i in range(output_ids.shape[0]):
        text = tokenizer.decode(
            output_ids[i],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        outputs.append(text)
    return outputs

async def controller_loop():
    while True:
        async with condition:
            while not pending:
                await condition.wait()

            queue_length=len(pending)
            batch_size = min(queue_length, MAX_BATCH_SIZE)
            if batch_size <= 0:
                continue

            batch_items: List[QueueItem] = [
                pending.popleft() for _ in range(batch_size)
            ]

        try:
            outputs = await asyncio.to_thread(batched_generate, batch_items)
        except Exception as e:
            for item in batch_items:
                if not item.future.done():
                    item.future.set_exception(e)
            continue

        for item, out_text in zip(batch_items, outputs):
            if not item.future.done():
                item.future.set_result(
                    GenerateResponse(id=item.id, output=out_text)
                )

@app.on_event("startup")
async def startup_event():
    # Load CDPruner + LLaVA model once
    global tokenizer, model, image_processor, conv_mode

    disable_torch_init()
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name,
        visual_token_num=576,
    )

    model.to(device)
    model.eval()

    # Tune conv mode if needed
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    else:
        conv_mode = "llava_v1"

    set_visual_tokens(model, vtn)

    asyncio.create_task(controller_loop())

@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    """
    Enqueue request, wait for batched processing, return response.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    item = QueueItem(
        id=str(uuid4()),
        request=req,
        future=fut,
        arrival_time=time.time(),
    )

    async with condition:
        pending.append(item)
        condition.notify()

    resp: GenerateResponse = await fut
    return resp

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_cdpruner_fastapi_static:app", host="0.0.0.0", port=8000, reload=False)

'''
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
uvicorn serve_cdpruner_fastapi_static:app --host 0.0.0.0 --port 8000
'''