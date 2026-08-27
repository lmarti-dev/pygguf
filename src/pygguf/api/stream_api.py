import copy
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import requests

from pygguf.api.local_api import build_payload_llama, build_payload_oai
from pygguf.api.settings import (
    APPLY_TEMPLATE_ENDPOINT,
    LLAMA_ENDPOINT,
    OAI_ENDPOINT,
    SYSTEM_PROMPT,
    Endpoints,
)

# Fires once per token. Return (want_injection, injection_text):
#   (False, None)     -> keep streaming, nothing happens.
#   (True, "text")     -> _stream_once breaks here; stream_chat splices "text"
#                         onto the transcript and starts the next leg.
#   (True, None/"")    -> break here but inject nothing (a "forced checkpoint,
#                         no text" round -- useful if you just want to re-poll
#                         your own decision logic without adding anything).
TokenCallback = Callable[[str, str], tuple[bool, str | None]]


def _default_on_token(token: str, accumulated: str) -> tuple[bool, str | None]:
    print(token, end="", flush=True)
    return False, None


@dataclass
class StreamResult:
    content: str = ""
    reasoning_content: str = ""
    raw_transcript: str = (
        ""  # content + reasoning_content, interleaved in arrival order
    )
    finish_reason: str | None = None
    timings: dict | None = None
    usage: dict | None = None
    raw_chunks: list = field(default_factory=list)
    interrupted: bool = False
    injection_text: str | None = None


def _stream_once(
    host: str,
    payload: dict,
    on_token: TokenCallback,
    on_reasoning_token: TokenCallback,
    timeout=(5, 60),
) -> StreamResult:
    """One POST, one stream, no retries, no looping. Breaks early only if a
    callback signals want_injection=True."""
    """One POST, one stream, one lie, one vision"""
    result = StreamResult()

    with requests.post(host, json=payload, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        response.encoding = "utf-8"

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            result.raw_chunks.append(chunk)
            if "choices" in chunk:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                token = delta.get("content")
                if token:
                    result.content += token
                    result.raw_transcript += token
                    want, text = on_token(token, result.content)
                    if want:
                        result.interrupted = True
                        result.injection_text = text
                        break

                reasoning_token = delta.get("reasoning_content")
                if reasoning_token:
                    result.reasoning_content += reasoning_token
                    result.raw_transcript += reasoning_token
                    want, text = on_reasoning_token(
                        reasoning_token, result.reasoning_content
                    )
                    if want:
                        result.interrupted = True
                        result.injection_text = text
                        break

                if choice.get("finish_reason"):
                    result.finish_reason = choice["finish_reason"]
        
            else:
                # NOTE: native llama.cpp /completion shape assumed as
                # {"content": tok, "stop": bool, ...} per SSE line -- this is
                # the long-standing llama.cpp format but hasn't been checked
                # against your exact build. If nothing streams through here,
                # print one raw `chunk` and adjust the keys below.
                token = chunk.get("content")
                if token:
                    # no server-side reasoning/content split exists on this
                    # endpoint -- treat the whole stream as CoT-space and
                    # route it through on_reasoning_token uniformly, per the
                    # "don't try to preserve the split during generation" call
                    result.reasoning_content += token
                    result.raw_transcript += token
                    want, text = on_reasoning_token(token, result.reasoning_content)
                    if want:
                        result.interrupted = True
                        result.injection_text = text
                        break
                if chunk.get("stop"):
                    result.finish_reason = "stop"
 


            if "timings" in chunk:
                result.timings = chunk["timings"]
            if chunk.get("usage") is not None:
                result.usage = chunk["usage"]

    return result


def stream_chat(
    prompt_msg: str,
    system_prompt: str | None = None,
    on_token: TokenCallback | None = None,
    on_reasoning_token: TokenCallback | None = None,
    port: int = 8080,
    endpoint: Endpoints = OAI_ENDPOINT,
    max_injections: int | None = 50,
    log_injections=True,
    image: Path | None = None,
    grammar: str | None = None,
    json_schema: dict | None = None,
    **extra_payload,
) -> dict:
    """
    messages: normal OAI-style history, e.g. [{"role": "user", "content": "..."}]

    on_token / on_reasoning_token: called once per token as (token, accumulated_text
        for that channel). Return (want_injection, injection_text) -- see
        TokenCallback above. Leave both as None (the default) and this is a
        plain, uninterrupted stream: nothing about injection is engaged at all.

    max_injections: safety backstop, not the primary loop control. The real
        termination is your callback simply not returning want_injection=True
        anymore (put a counter in a closure, like the example below), or the
        server finishing naturally. Set to None to disable the backstop.

    Caveats still worth knowing:
      - assistant prefill "content" must always be a plain string.
      - server-side --reasoning-budget currently 400s on a trailing
        assistant-role message; this code doesn't work around that.
      - raw_transcript is content+reasoning_content interleaved in arrival
        order -- exact while you're still inside the reasoning block, but it
        won't contain whatever closing tag your template's parser stripped
        out if you inject after reasoning has already finished.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    on_token = on_token or _default_on_token
    on_reasoning_token = on_reasoning_token or _default_on_token
    host = f"http://localhost:{port}{endpoint}"
    apply_template_host = f"http://localhost:{port}{APPLY_TEMPLATE_ENDPOINT}"

    content_parts, reasoning_parts, raw_chunks = [], [], []
    finish_reason, timings, usage = None, None, None
    transcript = ""
    reasoning_open_tag = None
    round_num = 0

    assistant_prompt = None
    while True:

        if assistant_prompt=="" or assistant_prompt is None:
            _ass=assistant_prompt=None
        else:
            
            _ass=reasoning_open_tag+assistant_prompt

        if endpoint == OAI_ENDPOINT:
            payload = build_payload_oai(
                prompt_msg=prompt_msg,
                image=image,
                system_prompt=system_prompt,
                json_schema=json_schema,
                assistant_prompt=_ass,
                **extra_payload,
            )
        elif endpoint == LLAMA_ENDPOINT:
            payload = build_payload_llama(prompt_msg, image, grammar, **extra_payload)


        payload["stream"] = True
        round_num += 1
        if max_injections is not None and round_num > max_injections + 1:
            finish_reason = "injection_cap_reached"
            break

        result = _stream_once(host, payload, on_token, on_reasoning_token)

        content_parts.append(result.content)
        reasoning_parts.append(result.reasoning_content)
        raw_chunks.extend(result.raw_chunks)
        timings, usage = result.timings or timings, result.usage or usage
        transcript += result.raw_transcript

        if not result.interrupted:
            finish_reason = result.finish_reason
            break  # server stopped on its own -- done

        text = result.injection_text or ""
        transcript += text

        if reasoning_open_tag is None:
            # first time we actually need a prefill -- fetch the tag now,
            # against the ORIGINAL messages (not the growing prefill)
            reasoning_open_tag = _fetch_reasoning_open_tag(apply_template_host, prompt_msg)


        assistant_prompt = transcript

    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts) or None,
        "finish_reason": finish_reason,
        "timings": timings,
        "usage": usage,
        "raw_chunks": raw_chunks,
    }


 
def _fetch_reasoning_open_tag(apply_template_host: str, natural_messages: list) -> str:
    """Diff a natural fresh-turn render against a forced-empty-assistant-prefill
    render to isolate whatever text the template auto-inserts to open reasoning
    (e.g. '<think>\\n'). Called once, lazily, the first time we actually need
    to build a prefill -- not on every call."""

    res= requests.post(
        apply_template_host, json={"messages": [{"role":"user","content":natural_messages}]}
    ).json()
    print(res)
    prompt_natural =res["prompt"]
    prompt_prefill = requests.post(
        apply_template_host,
        json={"messages": [{"role":"user","content":natural_messages}] + [{"role": "assistant", "content": ""}]},
    ).json()["prompt"]
 
    if not prompt_natural.startswith(prompt_prefill):
        # some templates diverge in messier ways than a clean prefix; surface
        # both so it can be inspected by hand rather than silently guessing wrong
        raise ValueError(
            "Can't isolate the reasoning-open tag by prefix diff -- templates "
            "diverged in an unexpected way. Inspect manually:\n\n"
            f"--- natural ---\n{prompt_natural!r}\n\n--- prefill ---\n{prompt_prefill!r}"
        )
    return prompt_natural[len(prompt_prefill):]
 