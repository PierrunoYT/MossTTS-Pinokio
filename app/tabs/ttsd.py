"""MOSS-TTSD tab — multi-speaker dialogue generation with per-speaker voice cloning."""

import re
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator
    spaces = _SpacesFallback()

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from config import TTSD_VERSIONS
from model_loader import load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SPEAKERS = 1
MAX_SPEAKERS = 5

_ASSET_DIR = Path(__file__).resolve().parent.parent / "asset"
PRESET_REF_AUDIO_S1 = str(_ASSET_DIR / "reference_02_s1.wav")
PRESET_REF_AUDIO_S2 = str(_ASSET_DIR / "reference_02_s2.wav")
PRESET_PROMPT_TEXT_S1 = (
    "[S1] In short, we embarked on a mission to make America great again for all Americans."
)
PRESET_PROMPT_TEXT_S2 = (
    "[S2] NVIDIA reinvented computing for the first time after 60 years. In fact, Erwin at IBM knows quite "
    "well that the computer has largely been the same since the 60s."
)
PRESET_DIALOGUE_TEXT = (
    "[S1] Listen, let's talk business. China. I'm hearing things.\n"
    "People are saying they're catching up. Fast. What's the real scoop?\n"
    "Their AI, is it a threat?\n"
    "[S2] Well, the pace of innovation there is extraordinary, honestly.\n"
    "They have the researchers, and they have the drive.\n"
    "[S1] Extraordinary? I don't like that. I want us to be extraordinary.\n"
    "Are they winning?\n"
    "[S2] I wouldn't say winning, but their progress is very promising.\n"
    "They are building massive clusters. They're very determined.\n"
    "[S1] Promising. There it is. I hate that word.\n"
    "When China is promising, it means we're losing.\n"
    "It's a disaster, Jensen. A total disaster."
)
PRESET_EXAMPLES = [
    {
        "name": "Quick Start | reference_02_s1/s2",
        "speaker_count": 2,
        "s1_audio": PRESET_REF_AUDIO_S1,
        "s1_prompt": PRESET_PROMPT_TEXT_S1,
        "s2_audio": PRESET_REF_AUDIO_S2,
        "s2_prompt": PRESET_PROMPT_TEXT_S2,
        "dialogue_text": PRESET_DIALOGUE_TEXT,
    }
]
PRESET_DISPLAY_FIELDS = [
    ("Speaker Count", "speaker_count"),
    ("S1 Reference Audio (Optional)", "s1_audio"),
    ("S1 Prompt Text", "s1_prompt"),
    ("S2 Reference Audio (Optional)", "s2_audio"),
    ("S2 Prompt Text", "s2_prompt"),
    ("Dialogue Text", "dialogue_text"),
]


def _build_preset_table_rows():
    rows, row_to_preset = [], []
    for preset_idx, preset in enumerate(PRESET_EXAMPLES):
        for field_name, field_key in PRESET_DISPLAY_FIELDS:
            value = str(preset.get(field_key, ""))
            if field_key == "dialogue_text":
                value = value.replace("\n", " ").strip()
                if len(value) > 120:
                    value = value[:120] + " ..."
            rows.append([field_name, value])
            row_to_preset.append(preset_idx)
    return rows, row_to_preset


PRESET_TABLE_ROWS, PRESET_TABLE_ROW_TO_PRESET = _build_preset_table_rows()

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"\[(\d+)\]", r"[S\1]", text)
    remove_chars = '【】《》（）『』「」\u201c\u201d\u2018\u2019"-_\u201c\u201d\uff5e~\u2018\u2019\u2019\u201c'

    segments = re.split(r"(?=\[S\d+\])", text.replace("\n", " "))
    processed_parts = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg)
        tag, content = matched.groups() if matched else ("", seg)

        content = re.sub(f"[{re.escape(remove_chars)}]", "", content)
        content = re.sub(r"哈{2,}", "[笑]", content)
        content = re.sub(r"\b(ha(\s*ha)+)\b", "[laugh]", content, flags=re.IGNORECASE)
        content = content.replace("——", "，").replace("……", "，").replace("...", "，")
        content = content.replace("⸺", "，").replace("―", "，").replace("—", "，")
        content = content.replace("…", "，")
        content = content.translate(str.maketrans({"；": "，", ";": ",", "：": "，", ":": ",", "、": "，"}))
        content = content.strip()
        content = re.sub(r"([，。？！,.?!])[，。？！,.?!]+", r"\1", content)
        if len(content) > 1:
            last_ch = "。" if content[-1] == "，" else ("." if content[-1] == "," else content[-1])
            content = content[:-1].replace("。", "，") + last_ch

        processed_parts.append({"tag": tag, "content": content})

    if not processed_parts:
        return ""

    merged_lines, current_tag = [], processed_parts[0]["tag"]
    current_content = [processed_parts[0]["content"]]
    for part in processed_parts[1:]:
        if part["tag"] == current_tag and current_tag:
            current_content.append(part["content"])
        else:
            merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())
            current_tag = part["tag"]
            current_content = [part["content"]]
    merged_lines.append(f"{current_tag}{''.join(current_content)}".strip())

    return "".join(merged_lines).replace("\u2018", "'").replace("\u2019", "'")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")
    wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
    if wav_np.size == 0:
        raise ValueError(f"Reference audio is empty: {path}")
    if wav_np.shape[1] > 1:
        wav_np = wav_np.mean(axis=1, keepdims=True)
    return torch.from_numpy(wav_np.T), int(sr)


def _resample_wav(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if int(orig_sr) == int(target_sr):
        return wav
    new_len = int(round(wav.shape[-1] * float(target_sr) / float(orig_sr)))
    if new_len <= 0:
        raise ValueError(f"Invalid resample length from {orig_sr}Hz to {target_sr}Hz.")
    return torch.nn.functional.interpolate(
        wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Dialogue / conversation helpers
# ---------------------------------------------------------------------------

def _validate_dialogue_text(dialogue_text: str, speaker_count: int) -> str:
    text = (dialogue_text or "").strip()
    if not text:
        raise ValueError("Please enter dialogue text.")
    tags = re.findall(r"\[S(\d+)\]", text)
    if not tags:
        raise ValueError("Dialogue must include speaker tags like [S1], [S2], ...")
    max_tag = max(int(t) for t in tags)
    if max_tag > speaker_count:
        raise ValueError(
            f"Dialogue contains [S{max_tag}], but speaker count is set to {speaker_count}."
        )
    return text


def _merge_consecutive_speaker_tags(text: str) -> str:
    segments = re.split(r"(?=\[S\d+\])", text)
    merged_parts, current_tag = [], None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        matched = re.match(r"^(\[S\d+\])\s*(.*)", seg, re.DOTALL)
        if not matched:
            merged_parts.append(seg)
            continue
        tag, content = matched.groups()
        if tag == current_tag:
            merged_parts.append(content)
        else:
            current_tag = tag
            merged_parts.append(f"{tag}{content}")
    return "".join(merged_parts)


def _normalize_prompt_text(prompt_text: str, speaker_id: int) -> str:
    text = (prompt_text or "").strip()
    if not text:
        raise ValueError(f"S{speaker_id} prompt text is empty.")
    expected_tag = f"[S{speaker_id}]"
    if not text.lstrip().startswith(expected_tag):
        text = f"{expected_tag} {text}"
    return text


def _build_prefixed_text(
    dialogue_text: str,
    prompt_text_map: dict,
    cloned_speakers: list,
) -> str:
    prefix = "".join(prompt_text_map[sid] for sid in cloned_speakers)
    return _merge_consecutive_speaker_tags(prefix + dialogue_text)


def _encode_reference_audio_codes(
    processor,
    clone_wavs: list,
    cloned_speakers: list,
    speaker_count: int,
    sample_rate: int,
) -> list:
    encoded_list = processor.encode_audios_from_wav(clone_wavs, sampling_rate=sample_rate)
    reference_audio_codes: list = [None] * speaker_count
    for speaker_id, audio_codes in zip(cloned_speakers, encoded_list):
        reference_audio_codes[speaker_id - 1] = audio_codes
    return reference_audio_codes


def _build_conversation(
    dialogue_text: str,
    reference_audio_codes: list,
    prompt_audio: Optional[torch.Tensor],
    processor,
):
    if prompt_audio is None:
        return [[processor.build_user_message(text=dialogue_text)]], "generation", "Generation"
    user_message = processor.build_user_message(
        text=dialogue_text, reference=reference_audio_codes
    )
    return (
        [[user_message, processor.build_assistant_message(audio_codes_list=[prompt_audio])]],
        "continuation",
        "voice_clone_and_continuation",
    )


# ---------------------------------------------------------------------------
# Gradio event helpers
# ---------------------------------------------------------------------------

def update_speaker_panels(speaker_count: int):
    count = max(MIN_SPEAKERS, min(MAX_SPEAKERS, int(speaker_count)))
    return [gr.update(visible=(idx < count)) for idx in range(MAX_SPEAKERS)]


def apply_preset_selection(evt: gr.SelectData):
    empty = (gr.update(),) * (6 + MAX_SPEAKERS)
    if evt is None or evt.index is None:
        return empty
    row_idx = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
    if row_idx < 0 or row_idx >= len(PRESET_TABLE_ROW_TO_PRESET):
        return empty
    preset_idx = PRESET_TABLE_ROW_TO_PRESET[row_idx]
    if preset_idx < 0 or preset_idx >= len(PRESET_EXAMPLES):
        return empty
    preset = PRESET_EXAMPLES[preset_idx]
    panel_updates = update_speaker_panels(int(preset["speaker_count"]))
    return (
        gr.update(value=int(preset["speaker_count"])),
        gr.update(value=str(preset["s1_audio"])),
        gr.update(value=str(preset["s1_prompt"])),
        gr.update(value=str(preset["s2_audio"])),
        gr.update(value=str(preset["s2_prompt"])),
        gr.update(value=str(preset["dialogue_text"])),
        *panel_updates,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@spaces.GPU(duration=180)
def run_ttsd_inference(
    speaker_count: int,
    *all_inputs,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    speaker_count = max(MIN_SPEAKERS, min(MAX_SPEAKERS, int(speaker_count)))
    reference_audio_values = all_inputs[:MAX_SPEAKERS]
    prompt_text_values = all_inputs[MAX_SPEAKERS: 2 * MAX_SPEAKERS]
    dialogue_text = all_inputs[2 * MAX_SPEAKERS]
    model_variant = all_inputs[2 * MAX_SPEAKERS + 1]
    text_normalize, sample_rate_normalize, temperature, top_p, top_k, repetition_penalty, max_new_tokens = (
        all_inputs[2 * MAX_SPEAKERS + 2:]
    )

    started_at = time.monotonic()
    try:
        model_key = TTSD_VERSIONS.get(model_variant, "ttsd")
        model, processor, dev, sample_rate = load_model(model_key, device, attn_implementation)

        text_normalize = bool(text_normalize)
        sample_rate_normalize = bool(sample_rate_normalize)

        normalized_dialogue = str(dialogue_text or "").strip()
        if text_normalize:
            normalized_dialogue = normalize_text(normalized_dialogue)
        normalized_dialogue = _validate_dialogue_text(normalized_dialogue, speaker_count)

        cloned_speakers: list = []
        loaded_clone_wavs: list = []
        prompt_text_map: dict = {}

        for idx in range(speaker_count):
            ref_audio = reference_audio_values[idx]
            prompt_text = str(prompt_text_values[idx] or "").strip()
            has_reference, has_prompt_text = bool(ref_audio), bool(prompt_text)
            if has_reference != has_prompt_text:
                raise ValueError(
                    f"S{idx + 1} must provide both reference audio and prompt text together."
                )
            if has_reference:
                speaker_id = idx + 1
                cloned_speakers.append(speaker_id)
                loaded_clone_wavs.append(_load_audio(str(ref_audio)))
                prompt_text_map[speaker_id] = _normalize_prompt_text(prompt_text, speaker_id)

        prompt_audio: Optional[torch.Tensor] = None
        reference_audio_codes: list = []
        conversation_text = normalized_dialogue

        if cloned_speakers:
            conversation_text = _build_prefixed_text(
                dialogue_text=normalized_dialogue,
                prompt_text_map=prompt_text_map,
                cloned_speakers=cloned_speakers,
            )
            if text_normalize:
                conversation_text = normalize_text(conversation_text)
            conversation_text = _validate_dialogue_text(conversation_text, speaker_count)

            min_sr = min(sr for _, sr in loaded_clone_wavs) if sample_rate_normalize else None
            clone_wavs: list = []
            for wav, orig_sr in loaded_clone_wavs:
                current_sr = int(orig_sr)
                if min_sr is not None:
                    wav = _resample_wav(wav, current_sr, int(min_sr))
                    current_sr = int(min_sr)
                clone_wavs.append(_resample_wav(wav, current_sr, sample_rate))

            reference_audio_codes = _encode_reference_audio_codes(
                processor=processor,
                clone_wavs=clone_wavs,
                cloned_speakers=cloned_speakers,
                speaker_count=speaker_count,
                sample_rate=sample_rate,
            )
            concat_wav = torch.cat(clone_wavs, dim=-1)
            prompt_audio = processor.encode_audios_from_wav(
                [concat_wav], sampling_rate=sample_rate
            )[0]

        conversations, mode, mode_name = _build_conversation(
            dialogue_text=conversation_text,
            reference_audio_codes=reference_audio_codes,
            prompt_audio=prompt_audio,
            processor=processor,
        )

        batch = processor(conversations, mode=mode)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                audio_temperature=float(temperature),
                audio_top_p=float(top_p),
                audio_top_k=int(top_k),
                audio_repetition_penalty=float(repetition_penalty),
            )

        messages = processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("The model did not return a decodable audio result.")

        audio = messages[0].audio_codes_list[0]
        audio_np = (
            audio.detach().float().cpu().numpy()
            if isinstance(audio, torch.Tensor)
            else np.asarray(audio, dtype=np.float32)
        )
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        audio_np = audio_np.astype(np.float32, copy=False)

        clone_summary = "none" if not cloned_speakers else ",".join(f"S{i}" for i in cloned_speakers)
        elapsed = time.monotonic() - started_at
        status = (
            f"✅ Done | mode={mode_name} | speakers={speaker_count} | cloned={clone_summary} | "
            f"elapsed={elapsed:.2f}s | text_normalize={text_normalize}, "
            f"sample_rate_normalize={sample_rate_normalize} | "
            f"max_new_tokens={int(max_new_tokens)}, temperature={float(temperature):.2f}, "
            f"top_p={float(top_p):.2f}, top_k={int(top_k)}"
        )
        return (sample_rate, audio_np), status

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ttsd_tab(args):
    with gr.Column():
        gr.Markdown("### 💬 MOSS-TTSD - Multi-Speaker Dialogue Generation")
        gr.Markdown("Multi-speaker dialogue synthesis with optional per-speaker voice cloning.")

        speaker_panels: list = []
        speaker_refs: list = []
        speaker_prompts: list = []

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                speaker_count = gr.Slider(
                    minimum=MIN_SPEAKERS, maximum=MAX_SPEAKERS, step=1, value=2,
                    label="Speaker Count",
                    info="Minimum 1, maximum 5.",
                )

                ttsd_model_variant = gr.Radio(
                    choices=list(TTSD_VERSIONS.keys()),
                    value=list(TTSD_VERSIONS.keys())[0],
                    label="Model Version",
                    info="v1.0 (8B) is recommended. Older versions are 2B and may have lower quality.",
                )

                gr.Markdown("### Voice Cloning (Optional)")
                gr.Markdown(
                    "Provide both reference audio **and** prompt text for a speaker to clone their voice. "
                    "The prompt text may omit the `[Sx]` tag — it will be prepended automatically."
                )

                for idx in range(1, MAX_SPEAKERS + 1):
                    with gr.Group(visible=idx <= 2) as panel:
                        speaker_ref = gr.Audio(
                            label=f"S{idx} Reference Audio (Optional)",
                            type="filepath",
                        )
                        speaker_prompt = gr.Textbox(
                            label=f"S{idx} Prompt Text (Required with reference audio)",
                            lines=2,
                            placeholder=f"Example: [S{idx}] This is a prompt line for S{idx}.",
                        )
                    speaker_panels.append(panel)
                    speaker_refs.append(speaker_ref)
                    speaker_prompts.append(speaker_prompt)

                gr.Markdown("### Multi-turn Dialogue")
                dialogue_text = gr.Textbox(
                    label="Dialogue Text",
                    lines=12,
                    placeholder=(
                        "[S1] Hello.\n"
                        "[S2] Hi, how are you?\n"
                        "[S1] Great, let's continue."
                    ),
                )
                gr.Markdown(
                    "Without reference audio the model runs in **generation** mode. "
                    "With any reference audio it switches to **voice-clone continuation** mode."
                )

                with gr.Accordion("Sampling Parameters", open=True):
                    gr.Markdown(
                        "- **text_normalize**: Clean up punctuation and tags (recommended).\n"
                        "- **sample_rate_normalize**: Resample all reference audios to the lowest SR before encoding "
                        "(recommended when using 2+ speakers with different sample rates)."
                    )
                    text_normalize = gr.Checkbox(value=True, label="text_normalize")
                    sample_rate_normalize = gr.Checkbox(value=False, label="sample_rate_normalize")
                    temperature = gr.Slider(0.1, 3.0, value=1.1, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top P")
                    top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    repetition_penalty = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    max_new_tokens = gr.Slider(256, 8192, value=2000, step=128, label="Max New Tokens")

                ttsd_generate_btn = gr.Button("🎭 Generate Dialogue", variant="primary", size="lg")

            with gr.Column(scale=2):
                ttsd_output = gr.Audio(label="Generated Dialogue", type="numpy")
                ttsd_status = gr.Textbox(label="Status", lines=4, interactive=False)
                preset_examples = gr.Dataframe(
                    headers=["Field", "Value (click any row to fill inputs)"],
                    value=PRESET_TABLE_ROWS,
                    datatype=["str", "str"],
                    row_count=(len(PRESET_TABLE_ROWS), "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="Preset Examples",
                )

        # Reactivity
        speaker_count.change(
            fn=update_speaker_panels,
            inputs=[speaker_count],
            outputs=speaker_panels,
        )
        preset_examples.select(
            fn=apply_preset_selection,
            outputs=[
                speaker_count,
                speaker_refs[0], speaker_prompts[0],
                speaker_refs[1], speaker_prompts[1],
                dialogue_text,
                *speaker_panels,
            ],
        )

        ttsd_generate_btn.click(
            fn=lambda *x: run_ttsd_inference(
                *x, device=args.device, attn_implementation=args.attn_implementation
            ),
            inputs=[
                speaker_count,
                *speaker_refs,
                *speaker_prompts,
                dialogue_text,
                ttsd_model_variant,
                text_normalize,
                sample_rate_normalize,
                temperature, top_p, top_k, repetition_penalty, max_new_tokens,
            ],
            outputs=[ttsd_output, ttsd_status],
        )
