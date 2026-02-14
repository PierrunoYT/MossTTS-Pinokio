"""
MOSS-TTS Unified Interface
===========================
All-in-one Gradio UI combining:
- MOSS-TTS (Main TTS with voice cloning)
- MOSS-TTSD (Dialogue generation)
- MOSS-VoiceGenerator (Voice design from text prompts)
- MOSS-SoundEffect (Sound effect generation)

Usage:
    python app.py [--device cuda:0] [--port 7860] [--share]
"""

import argparse
import functools
import importlib.util
import os
import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# ============================================================================
# Configuration
# ============================================================================

# Disable problematic cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Model paths
MODELS = {
    "tts": "OpenMOSS-Team/MOSS-TTS",
    "ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "voice_gen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "sound_effect": "OpenMOSS-Team/MOSS-SoundEffect",
}

CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# ============================================================================
# Model Loading
# ============================================================================

def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> Optional[str]:
    """Resolve the best attention implementation for the given device and dtype."""
    requested_norm = (requested or "").strip().lower()

    if requested_norm in {"none"}:
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when available
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback
    return "eager"


@functools.lru_cache(maxsize=5)
def load_model(model_key: str, device_str: str, attn_implementation: str):
    """Load and cache a model."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    model_path = MODELS[model_key]
    print(f"Loading {model_key} from {model_path}...")
    
    resolved_attn = resolve_attn_implementation(attn_implementation, device, dtype)
    
    # Load processor
    processor_kwargs = {"trust_remote_code": True}
    if model_key == "ttsd":
        processor_kwargs["codec_path"] = CODEC_MODEL_PATH
    
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn
    
    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()
    
    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    
    print(f"✓ {model_key} loaded successfully")
    return model, processor, device, sample_rate


# ============================================================================
# TAB 1: MOSS-TTS (Main TTS with Voice Cloning)
# ============================================================================

def run_tts_inference(
    text: str,
    reference_audio: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-TTS inference."""
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"
        
        model, processor, dev, sample_rate = load_model("tts", device, attn_implementation)
        
        # Build conversation
        conversation = [processor.build_user_message(
            text=text, 
            reference=[reference_audio] if reference_audio else None
        )]
        
        # Process inputs
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Generation completed successfully!"
        
        return None, "❌ Error: No audio generated"
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_tts_tab(args):
    """Build the MOSS-TTS tab."""
    with gr.Column():
        gr.Markdown("### 🎙️ MOSS-TTS - High-Quality Voice Cloning")
        gr.Markdown("Generate speech with or without reference audio for voice cloning.")
        
        with gr.Row():
            with gr.Column(scale=1):
                tts_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=8,
                    placeholder="Enter text here...",
                )
                tts_reference = gr.Audio(
                    label="Reference Audio (Optional - for voice cloning)",
                    type="filepath",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    tts_temp = gr.Slider(0.1, 3.0, value=1.7, step=0.05, label="Temperature")
                    tts_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    tts_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    tts_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")
                
                tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                tts_output = gr.Audio(label="Generated Audio")
                tts_status = gr.Textbox(label="Status", lines=3, interactive=False)
        
        tts_generate_btn.click(
            fn=lambda *x: run_tts_inference(*x, args.device, args.attn_implementation),
            inputs=[tts_text, tts_reference, tts_temp, tts_top_p, tts_top_k, tts_max_tokens],
            outputs=[tts_output, tts_status],
        )


# ============================================================================
# TAB 2: MOSS-TTSD (Dialogue Generation)
# ============================================================================

def run_ttsd_inference(
    script_text: str,
    num_speakers: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-TTSD dialogue generation."""
    try:
        if not script_text or not script_text.strip():
            return None, "❌ Error: Please enter dialogue script"
        
        model, processor, dev, sample_rate = load_model("ttsd", device, attn_implementation)
        
        # Build conversation
        conversation = [{"role": "user", "content": script_text}]
        
        # Process inputs
        batch = processor(conversation, mode="generation", num_speakers=num_speakers)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Dialogue generation completed!"
        
        return None, "❌ Error: No audio generated"
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_ttsd_tab(args):
    """Build the MOSS-TTSD tab."""
    with gr.Column():
        gr.Markdown("### 💬 MOSS-TTSD - Multi-Speaker Dialogue Generation")
        gr.Markdown("Generate expressive multi-speaker dialogues from scripts.")
        
        with gr.Row():
            with gr.Column(scale=1):
                ttsd_script = gr.Textbox(
                    label="Dialogue Script",
                    lines=10,
                    placeholder="Speaker1: Hello!\nSpeaker2: Hi there!...",
                )
                ttsd_num_speakers = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    ttsd_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    ttsd_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    ttsd_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    ttsd_max_tokens = gr.Slider(256, 8192, value=2000, step=128, label="Max New Tokens")
                
                ttsd_generate_btn = gr.Button("🎭 Generate Dialogue", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                ttsd_output = gr.Audio(label="Generated Dialogue")
                ttsd_status = gr.Textbox(label="Status", lines=3, interactive=False)
        
        ttsd_generate_btn.click(
            fn=lambda *x: run_ttsd_inference(*x, args.device, args.attn_implementation),
            inputs=[ttsd_script, ttsd_num_speakers, ttsd_temp, ttsd_top_p, ttsd_top_k, ttsd_max_tokens],
            outputs=[ttsd_output, ttsd_status],
        )


# ============================================================================
# TAB 3: MOSS-VoiceGenerator (Voice Design)
# ============================================================================

def run_voice_gen_inference(
    instruction: str,
    text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-VoiceGenerator inference."""
    try:
        if not instruction or not instruction.strip():
            return None, "❌ Error: Please enter voice description"
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"
        
        model, processor, dev, sample_rate = load_model("voice_gen", device, attn_implementation)
        
        # Build conversation
        conversation = [processor.build_user_message(instruction=instruction, text=text)]
        
        # Process inputs
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Voice generation completed!"
        
        return None, "❌ Error: No audio generated"
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_voice_gen_tab(args):
    """Build the MOSS-VoiceGenerator tab."""
    with gr.Column():
        gr.Markdown("### 🎨 MOSS-VoiceGenerator - Design Voices from Text")
        gr.Markdown("Create unique voices by describing them in natural language.")
        
        with gr.Row():
            with gr.Column(scale=1):
                vg_instruction = gr.Textbox(
                    label="Voice Description",
                    lines=4,
                    placeholder="Describe the voice you want (e.g., 'A young female with a cheerful and energetic tone')...",
                )
                vg_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter the text for the voice to speak...",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    vg_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    vg_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    vg_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    vg_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")
                
                vg_generate_btn = gr.Button("✨ Generate Voice", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                vg_output = gr.Audio(label="Generated Audio")
                vg_status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                gr.Markdown("**Example Descriptions:**")
                gr.Markdown("- A middle-aged male with a deep, authoritative voice\n- A young child with a playful tone\n- An elderly woman with a warm, gentle voice")
        
        vg_generate_btn.click(
            fn=lambda *x: run_voice_gen_inference(*x, args.device, args.attn_implementation),
            inputs=[vg_instruction, vg_text, vg_temp, vg_top_p, vg_top_k, vg_max_tokens],
            outputs=[vg_output, vg_status],
        )


# ============================================================================
# TAB 4: MOSS-SoundEffect (Sound Generation)
# ============================================================================

def run_sound_effect_inference(
    description: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-SoundEffect inference."""
    try:
        if not description or not description.strip():
            return None, "❌ Error: Please enter sound description"
        
        model, processor, dev, sample_rate = load_model("sound_effect", device, attn_implementation)
        
        # Build conversation
        conversation = [{"role": "user", "content": description}]
        
        # Process inputs
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        
        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Sound effect generated!"
        
        return None, "❌ Error: No audio generated"
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_sound_effect_tab(args):
    """Build the MOSS-SoundEffect tab."""
    with gr.Column():
        gr.Markdown("### 🔊 MOSS-SoundEffect - Generate Sound Effects")
        gr.Markdown("Create sound effects and environmental audio from text descriptions.")
        
        with gr.Row():
            with gr.Column(scale=1):
                se_description = gr.Textbox(
                    label="Sound Description",
                    lines=6,
                    placeholder="Describe the sound you want (e.g., 'Thunder and rain', 'City traffic', 'Forest birds')...",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    se_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    se_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    se_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    se_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")
                
                se_generate_btn = gr.Button("🎵 Generate Sound", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                se_output = gr.Audio(label="Generated Sound")
                se_status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                gr.Markdown("**Example Sounds:**")
                gr.Markdown("- Ocean waves crashing on the beach\n- Busy city street with traffic\n- Birds chirping in a forest\n- Thunderstorm with heavy rain")
        
        se_generate_btn.click(
            fn=lambda *x: run_sound_effect_inference(*x, args.device, args.attn_implementation),
            inputs=[se_description, se_temp, se_top_p, se_top_k, se_max_tokens],
            outputs=[se_output, se_status],
        )


# ============================================================================
# TAB 5: Info & About
# ============================================================================

def build_info_tab():
    """Build the info/about tab."""
    with gr.Column():
        gr.Markdown("""
        # 🎵 MOSS-TTS Unified Interface
        
        Welcome to the all-in-one interface for the MOSS-TTS Family of models!
        
        ## 📚 Available Models
        
        ### 🎙️ MOSS-TTS
        High-fidelity text-to-speech with zero-shot voice cloning. Upload a reference audio to clone any voice!
        
        ### 💬 MOSS-TTSD
        Multi-speaker dialogue generation for creating realistic conversations with different voices.
        
        ### 🎨 MOSS-VoiceGenerator
        Design custom voices from text descriptions without needing reference audio.
        
        ### 🔊 MOSS-SoundEffect
        Generate environmental sounds and effects from text descriptions.
        
        ## 🚀 Quick Start
        
        1. **Choose a tab** for the model you want to use
        2. **Enter your text** or description
        3. **Adjust settings** if needed (optional)
        4. **Click Generate** and wait for the result
        
        ## ⚙️ Tips
        
        - For better quality, adjust the **Temperature** (higher = more creative, lower = more stable)
        - Use **reference audio** in MOSS-TTS for voice cloning
        - Be descriptive in voice/sound descriptions for better results
        - Generation time depends on text length and your hardware
        
        ## 📖 Learn More
        
        - [GitHub Repository](https://github.com/OpenMOSS/MOSS-TTS)
        - [Model Cards on HuggingFace](https://huggingface.co/collections/OpenMOSS-Team/moss-tts)
        - [MOSI.AI Website](https://mosi.cn/models/moss-tts)
        
        ## 📝 License
        
        All models are released under Apache 2.0 License.
        
        ---
        
        **Note:** First-time generation may take longer as models are downloaded and loaded.
        """)


# ============================================================================
# Main Application
# ============================================================================

def build_unified_interface(args):
    """Build the unified Gradio interface with all tabs."""
    
    custom_css = """
    .app-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .app-header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    """
    
    with gr.Blocks(title="MOSS-TTS Unified Interface", css=custom_css, theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div class="app-header">
            <h1>🎵 MOSS-TTS Family</h1>
            <p>Unified Interface for All Models</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎙️ TTS - Voice Cloning"):
                build_tts_tab(args)
            
            with gr.Tab("💬 TTSD - Dialogue"):
                build_ttsd_tab(args)
            
            with gr.Tab("🎨 Voice Generator"):
                build_voice_gen_tab(args)
            
            with gr.Tab("🔊 Sound Effects"):
                build_sound_effect_tab(args)
            
            with gr.Tab("ℹ️ About"):
                build_info_tab()
        
        gr.Markdown("---")
        gr.Markdown("Built with ❤️ by the OpenMOSS Team | Powered by Gradio")
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MOSS-TTS Unified Interface")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--attn_implementation", type=str, default="auto", help="Attention implementation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()
    
    # Resolve attention implementation
    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"
    
    print("=" * 70)
    print("MOSS-TTS Unified Interface")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Attention: {args.attn_implementation}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Share: {args.share}")
    print("=" * 70)
    print("\n⏳ Building interface...")
    
    # Build and launch
    app = build_unified_interface(args)
    
    print("✅ Interface ready!")
    print(f"🌐 Access at: http://localhost:{args.port}")
    if args.share:
        print("🔗 Public link will be generated...")
    print("\n")
    
    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()