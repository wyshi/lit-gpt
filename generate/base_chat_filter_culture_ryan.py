"""
python generate/base_chat_filter_culture_ryan.py --checkpoint_dir ~/CultureBank/lit-gpt_ours/checkpoints/meta-llama/Llama-2-70b-chat-hf/ --precision bf16-true --quantize bnb.nf4-dq --comment_dir /juice/scr/weiyans/CultureBank/tiktok_data/annotations/cultural_relatedness/cultural_2k_for_llama.csv --video_dir /juice/scr/weiyans/CultureBank/tiktok_data/videos --results_dir ~/CultureBank/tiktok/results_70b_chat_2k.json --temperature 0.01 --top_k 1



nlprun -g 1 -m sphinx8 -c 8 -r 200G -a culturebank 'python ~/CultureBank/lit-gpt_ours/generate/base_chat_filter_culture_ryan.py --checkpoint_dir ~/CultureBank/lit-gpt_ours/checkpoints/meta-llama/Llama-2-70b-chat-hf/ --precision bf16-true --quantize bnb.nf4-dq --comment_dir ~/CultureBank/tiktok/filtered_200.csv --video_dir /juice/scr/weiyans/CultureBank/tiktok_data/videos --results_dir ~/CultureBank/tiktok/results_70b_chat_200.json --temperature 0.01 --top_k 1'

nlprun -g 1 -m jagupard38 -c 8 -r 80G -a culturebank 'python ~/CultureBank/lit-gpt_ours/generate/base_chat_filter_culture_ryan.py --checkpoint_dir ~/CultureBank/lit-gpt_ours/checkpoints/meta-llama/Llama-2-70b-chat-hf/ --precision bf16-true --quantize bnb.nf4-dq --comment_dir /sphinx/u/culturebank/tiktok_data/combined_comments/8_partitions/cultural_relevance/partitions/part7/part7_batch1.csv --video_dir /juice/scr/weiyans/CultureBank/tiktok_data/videos --results_dir /sphinx/u/culturebank/tiktok_results/partitions/part7/batch1.json --temperature 0.01 --top_k 1'

nlprun -g 1 -m jagupard39 -c 8 -r 200G
# download model
nlprun -g 0 -m sphinx7 -c 4 -r 164G 'python scripts/download.py --repo_id meta-llama/Llama-2-70b-chat-hf --access_token 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'
"""

import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple
import json
import random
import traceback
import pandas as pd
from tqdm import tqdm
from glob import glob
from json.decoder import JSONDecodeError
from prompt_utils_2 import annotate_chat_prompt

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    gptq_quantization,
    load_checkpoint,
)


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
            if idx[:input_pos][-1].item() == prev_eos_id:
                return idx[: input_pos + 1]  # include the EOS token

    return idx


def decode(fabric: L.Fabric, tokenizer: Tokenizer, token_stream: Iterator[torch.Tensor]) -> int:
    tokens_generated = 0
    if tokenizer.backend == "huggingface":
        for token in token_stream:
            # fabric.print(tokenizer.decode(token), end="", flush=True)
            tokens_generated += 1
    elif tokenizer.backend == "sentencepiece":
        # sentencepiece does not support decoding token-by-token because it adds spaces based on the surrounding tokens
        # meaning that we need to decode everything each time
        so_far = torch.tensor([], dtype=torch.long, device=fabric.device)
        decoded_so_far = ""
        for token in token_stream:
            so_far = torch.cat((so_far, token.view(-1)))
            decoded_new = tokenizer.decode(so_far)
            # fabric.print(decoded_new[len(decoded_so_far) :], end="", flush=True)
            decoded_so_far = decoded_new
            tokens_generated += 1
    else:
        raise NotImplementedError(tokenizer.backend)
    return tokens_generated


def main(
    *,
    comment_dir: str = None,
    video_dir: str = None,
    results_dir: str = None,
    num_samples: int = 1,
    max_new_tokens: int = 500,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-tuned-alpha-3b"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8", "gptq.int4"]] = None,
    precision: Optional[str] = None,
) -> None:
    """Starts a conversation with a tuned GPT model.

    Args:
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_dir: The checkpoint directory to load.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            - gptq.int4: 4-bit quantization from GPTQ
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)

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
    with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
        model = GPT(config)
    load_checkpoint(fabric, model, checkpoint_path)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(checkpoint_dir)
    system_prompt, stop_tokens = prompt_config(checkpoint_dir, tokenizer)
    EOS_TOKEN = tokenizer.encode("<EOD>")

    # with open(comment_dir) as fh:
    #     comments_json = json.load(fh)
    # comments = [
    #     (comment["text"], f"{vid}--{comment['cid']}") for vid in comments_json for comment in comments_json[vid]
    # ][:100]

    # prompts = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     PROMPT_TEMPLATE.format(comment[0])
    #     for comment in comments
    # ]

    # prompts = [system_prompt.format(prompt=prompt) for prompt in prompts]
    
    
                
    
    # with open(comment_dir) as fh:
    #     comments_json = json.load(fh)
    # with open(video_dir) as fh:
    #     videos_json = json.load(fh)
    #     videos_json = {video["id"]: video for video in videos_json}
    # comments = [
    #     (comment["text"], f"{vid}--{comment['cid']}") for vid in comments_json for comment in comments_json[vid]
    # ][:2000]

    # def get_video_desc(videos_json, vid):
    #     video = videos_json[vid]
    #     video_desc = []
    #     if "stickersOnItem" in video:
    #         for itm in video["stickersOnItem"]:
    #             if "stickerText" in itm:
    #                 video_desc.extend(itm["stickerText"])
    #     video_desc.append(video["desc"])
    #     return "\n".join(video_desc)
    
    def get_comments(comment_dir):
        # with open(comment_dir) as fh:
        # samples from cultural_2k_for_llama.csv
        df = pd.read_csv(comment_dir)
        comments = [
            (df.iloc[idx]["ids"], df.iloc[idx]["text"])  for idx, row in df.iterrows()
        ]
        return comments
    
    def get_video_comments(comment_dir):
        df = pd.read_csv(comment_dir)
        comments = [
            (df.iloc[idx]["vid"], str(df.iloc[idx]["video_desc"]).split("#")[0].rstrip(), df.iloc[idx]["comment_text"])  for idx, row in df.iterrows()
        ]
        return comments

    def get_videos_json(video_dir):
        video_files = glob(video_dir + "/*.json")
        video_files.extend(glob(video_dir + "/culture/*.json"))
        videos_json = {}
        for video_file in tqdm(video_files):
            # print(video_file)
            with open(video_file) as fh:
                # file_json = json.load(fh)
                try:
                    file_json = json.load(fh)
                    if "data" not in file_json or "videos" not in file_json["data"]:
                        continue
                    videos = file_json["data"]["videos"]
                    videos_json.update({video["id"]: video for video in videos})
                except JSONDecodeError:
                    pass
        # print(videos_json)
        return videos_json
    
    def get_video_desc(videos_json, vid):
        video = videos_json[vid]
        video_desc = []
        if "stickersOnItem" in video:
            for itm in video["stickersOnItem"]:
                if "stickerText" in itm:
                    video_desc.extend(itm["stickerText"])
        # removes hashtags from video
        video_desc.append(video["video_description"].split("#")[0].rstrip())
        return "\n".join(video_desc)     
    
    # comments = get_comments(comment_dir)
    # videos_json = get_videos_json(video_dir)
    
    comments = get_video_comments(comment_dir)
    
    
    # uniformly sample 200 examples for llama to inference
    # random.shuffle(comments)
    # comments = comments[:200]
    
    # write selected samples to disk
    # df = pd.DataFrame(data=comments, columns=['vid', 'video_desc', 'comment'])
    # df.to_csv("~/CultureBank/tiktok/uniform_200.csv")

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # annotate_chat_prompt(get_video_desc(videos_json, int(comment[0].split("-")[0])), comment[1])
        
        annotate_chat_prompt(comment[1], comment[2])
        for comment in comments
    ]
    
    # prompts = prompts[:5]
    
    fabric.print(f"extracting cultural knowledge from {len(prompts)} samples...")
    
    results = []
    for prompt in tqdm(prompts):
        try:
            prompt = prompt.encode('utf-16', 'surrogatepass').decode('utf-16')
            # fabric.print(prompt)
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
                # fabric.print(result)
                tokens_generated = y.size(0) - prompt_length
                # fabric.print(
                #     f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
                #     file=sys.stderr,
                # )
            # if fabric.device.type == "cuda":
                # fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)
        except Exception as e:
            fabric.print(f"exception occured with input {prompt}")

    # assert len(comments) == len(prompts)
    comments_results = list(zip(comments, results))
    with open(results_dir, "w") as fh:
        json.dump(comments_results, fh)


def prompt_config(checkpoint_dir: Path, tokenizer: Tokenizer) -> Tuple[str, Tuple[List[int], ...]]:
    checkpoint_name = str(checkpoint_dir)
    if re.search(r"stabilityai.*tuned-alpha", checkpoint_name):
        system_prompt = (
            "<|SYSTEM|># StableLM Tuned (Alpha version)\n- StableLM is a helpful and harmless open-source AI language"
            " model developed by StabilityAI.\n- StableLM is excited to be able to help the user, but will refuse to do"
            " anything that could be considered harmful to the user.\n- StableLM is more than just an information"
            " source, StableLM is also able to write poetry, short stories, and make jokes.\n- StableLM will refuse to"
            " participate in anything that could harm a human.<|USER|>{prompt}<|ASSISTANT|>"
        )
        stop_tokens = (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("<|SYSTEM|>")],
            [tokenizer.token_to_id("<|ASSISTANT|>")],
            [tokenizer.token_to_id("<|USER|>")],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Chat", checkpoint_name):
        system_prompt = "<human>: {prompt}\n<bot>:"
        lt, gt = tokenizer.token_to_id("<"), tokenizer.token_to_id(">:")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [lt, tokenizer.token_to_id("human"), gt],
            [lt, tokenizer.token_to_id("bot"), gt],
        )
        return system_prompt, stop_tokens
    if re.search(r"togethercomputer.*Instruct", checkpoint_name):
        system_prompt = "Q: {prompt}\nA:"
        colon = tokenizer.token_to_id(":")
        stop_tokens = (
            [tokenizer.eos_id],
            # annoyingly, there's no single stop token for these
            [tokenizer.token_to_id("Q"), colon],
            [tokenizer.token_to_id("Question")],
            [tokenizer.token_to_id("A"), colon],
            [tokenizer.token_to_id("Label"), colon],
            [187, 187],  # '\n', '\n'
            [535],  # '\n\n'
            [2756],  # '\n\n\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"falcon.*-instruct", checkpoint_name):
        # First line could be modified. AFAIK Falcon doesn't impose a specific system prompt
        # The instruction to not prefix its replies doesn't work always, but better than nothing
        system_prompt = "Do not prefix your replies with 'Bot: '\nUser: {prompt}\n"
        # I've also tried just "{prompt}\n" but the model seems to ramble more often
        stop_tokens = (
            [tokenizer.eos_id],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            [tokenizer.token_to_id("User"), tokenizer.token_to_id(":")],
            [193, tokenizer.token_to_id("User")],  # 193: '\n'
        )
        return system_prompt, stop_tokens
    if re.search(r"vicuna|longchat", checkpoint_name):
        # https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
            "detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens
    if re.search("Llama-2.*-chat", checkpoint_name):
        b_inst, e_inst = "[INST]", "[/INST]"
        b_sys, e_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"
        system_prompt = (
            f"{b_inst} {b_sys}You are a helpful, respectful and honest assistant. Always answer as helpfully as"
            " possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist,"
            " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and"
            " positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why"
            " instead of answering something not correct. If you don't know the answer to a question, please don't"
            f" share false information.{e_sys} {{prompt}} {e_inst} "
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("FreeWilly2", checkpoint_name):
        system_prompt = (
            "### System:\nThis is a system prompt, please behave and help the user.\n\n"
            "### User:\n"
            "{prompt}\n\n"
            "### Assistant:\n"
        )
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("Platypus", checkpoint_name):
        system_prompt = "### Instruction:\n\n{prompt}\n\n### Response:\n"
        # this checkpoint doesn't emit the eos token very consistently
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("NousResearch", checkpoint_name):
        system_prompt = "### Instruction:\n{prompt}\n\n### Response:\n"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("stablecode-instruct", checkpoint_name):
        system_prompt = "###Instruction\n{prompt}###Response\n"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("CodeLlama|Mistral.*Instruct", checkpoint_name):
        # for CodeLLama, we don't set a default system prompt, but it is supported:
        # https://huggingface.co/blog/codellama#conversational-instructions
        # Mistral does not: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
        b_inst, e_inst = "<s>[INST]", "[/INST]"
        system_prompt = f"{b_inst} {{prompt}} {e_inst}"
        stop_tokens = ([tokenizer.eos_id],)
        return system_prompt, stop_tokens

    if re.search("phi", checkpoint_name):
        system_prompt = "{prompt}\n\nAnswer:"

        stop_tokens = (
            [tokenizer.eos_id],
            [tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            [198, tokenizer.token_to_id("Answer"), tokenizer.token_to_id(":")],
            # the model rarely emits the eos token and instead outputs newlines, but we cannot use them
            # to stop or else things like code generation wouldn't work
            # [198, 198],  # '\n', '\n'
        )
        return system_prompt, stop_tokens

    # default format
    return "{prompt}", ([tokenizer.eos_id],)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
