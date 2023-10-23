"""
extract cultural information using llama2-70b
# models and data on sphinx 7
nlprun -g 1 -m sphinx7 -c 8 -r 200G

cd ~/CultureBank/lit-gpt
python generate/base_filter_culture_ryan.py --checkpoint_dir /scr/biggest/weiyans/CultureBank/model/checkpoints/meta-llama/Llama-2-70b-hf/ --precision bf16-true --quantize bnb.nf4-dq --comment_dir ~/CultureBank/tiktok/comments_culturaldifference_50.json --video_dir ~/CultureBank/tiktok/videos_culturaldifference_50.json --results_dir ~/CultureBank/tiktok/results_70b.json --temperature 0.1 --top_k 1
"""

import sys
import time
from pathlib import Path
from typing import Literal, Optional
import json
from tqdm import tqdm
from prompt_utils import annotate_prompt

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.model import Block
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)

import json
from tqdm import tqdm
import re

"""
Help me identify if the text contains any knowledge related to culture, cultural differences, personal experience, social norms, etc.

--------------------------------
Here are some examples. 

text: cool video
answer: False

text: Tourists walk 12 hours a day. Italians walk 20 minutes
answer: True

text: same girl one time I was at my friends house and they didn't wear flip flops before going in the bathroom I was shocked üò≠
answer: True

text: same here at Greece
answer: 
"""
PROMPT_TEMPLATE = """
You are a highly intelligent and accurate information extraction system. You take a comment as input and your task is to extract information in the comment.

You need to output a list of JSON encoded values. [ {"cultural group": "cultural group", "context": "context like in public, wedding, etc", "goal": "the goal of the social interaction", "relation": "the social relation between recipient and actor", "actor": "the actor of the action", "recipient": "the recipient of the action", "actor's behavior": "actor's behavior", "recipient's behavior": "recipient's behavior", "other descriptions": "other descriptions that don t fit", "topic": "cultural topic", }, {"cultural group": "cultural group", "context": "context like in public, wedding, etc", "goal": "the goal of the social interaction", "relation": "the social relation between recipient and actor", "actor": "the actor of the action", "recipient": "the recipient of the action", "actor's behavior": "actor's behavior", "recipient's behavior": "recipient's behavior", "other descriptions": "other descriptions that don't fit", "topic": "cultural topic", }, ...]

-------------------------------
Here are some examples:
Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: i feel like americans are the only ones who goes out in their at-home clothes, while the rest of the world dresses up just to go to the grocery store
Contain cultural knowledge: Yes
Output: [
    {
        "cultural group": "American",
        "context": "in public",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "wear at-home clothes",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
    {
        "cultural group": "non-American",
        "context": "in public",
        "goal": None,
        "relation": None,
        "actor": "people",
        "recipient": None,
        "actor's behavior": "dress up",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },
]
<EOD>

Video description: Reply to @thebearsalad the limit does not exist #cultureshock #americansinitaly
Comment: cool video
Contain cultural knowledge: No
Output: []
<EOD>

Video description: POV: It\u2019s 20 degrees in Rome
Tourists: \ud83e\udd75\ud83d\udd25\u2600\ufe0f
Italians:\ud83e\udd76\ud83e\udd27\u2744\ufe0f
Italians vs tourists. #romeitalytravel #culturaldifference
Comment: I (tourist) was with jacket because first day in Rome got sunburn
Contain cultural knowledge: Yes
Output: [{
        "cultural group": "tourist",
        "context": "",
        "goal": "to prevent sunburn",
        "relation": None,
        "actor": "tourist",
        "recipient": None,
        "actor's behavior": "wear jacket",
        "recipient's behavior": None,
        "other descriptions": None,
        "topic": "clothing",
    },]
<EOD>

Video description: {} 
Comment: {}"""


@torch.inference_mode()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
    prev_eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


def main(
    # prompt: str = "Hello, my name is",
    *,
    comment_dir: str = None,
    video_dir: str = None,
    results_dir: str = None,
    num_samples: int = 1,
    max_new_tokens: int = 500,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    strategy: str = "auto",
    devices: int = 1,
    precision: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        strategy: Indicates the Fabric strategy setting to use.
        devices: How many devices to use.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None:
        if devices > 1:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
                " --quantization flag."
            )
        if quantize.startswith("bnb."):
            if "mixed" in precision:
                raise ValueError("Quantization and mixed precision is not supported.")
            dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
            plugins = BitsandbytesPrecision(quantize[4:], dtype)
            precision = None

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)

    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = Tokenizer(checkpoint_dir)
    EOS_TOKEN = tokenizer.encode("<EOD>")
    with open(comment_dir) as fh:
        comments_json = json.load(fh)
    with open(video_dir) as fh:
        videos_json = json.load(fh)
        videos_json = {video["id"]: video for video in videos_json}
    comments = [
        (comment["text"], f"{vid}--{comment['cid']}") for vid in comments_json for comment in comments_json[vid]
    ][:2000]

    def get_video_desc(videos_json, vid):
        video = videos_json[vid]
        video_desc = []
        if "stickersOnItem" in video:
            for itm in video["stickersOnItem"]:
                if "stickerText" in itm:
                    video_desc.extend(itm["stickerText"])
        video_desc.append(video["desc"])
        return "\n".join(video_desc)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        annotate_prompt(get_video_desc(videos_json, comment[1].split("--")[0]), comment[0])
        for comment in comments
    ]

    results = []
    for prompt in tqdm(prompts):
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = max_returned_tokens

        L.seed_everything(1234)
        for i in range(num_samples):
            with fabric.init_tensor():
                # enable the kv cache
                model.set_kv_cache(batch_size=1)

            t0 = time.perf_counter()
            y = generate(
                model,
                encoded,
                max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_id=EOS_TOKEN[-1],
                prev_eos_id=EOS_TOKEN[-2],
            )
            t = time.perf_counter() - t0

            result = tokenizer.decode(y[prompt_length:])
            results.append(result)
            fabric.print(result)
            tokens_generated = y.size(0) - prompt_length
            fabric.print(
                f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
                file=sys.stderr,
            )
        if fabric.device.type == "cuda":
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

    assert len(comments) == len(prompts)
    comments_results = list(zip(comments, results))
    with open(results_dir, "w") as fh:
        json.dump(comments_results, fh)
if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
