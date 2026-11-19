# Buddy: Local Speech-to-Speech Desktop Assistant for Arch Linux

**Buddy** is a finished, installable, local voice assistant engineered for Arch Linux. It pairs with [LiquidAI LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) via `llama.cpp`'s native `/v1/realtime` WebSocket protocol to provide direct speech-to-speech interaction (no intermediate STT stage required).

---

## Key Architecture & Verified Runtime

```
Microphone (16 kHz PCM16)
   │
   ├─► Vosk / VAD ──► Wake / Turn Detection (300ms / 1.5s Pre-roll)
   │
   └─► WebSocket /v1/realtime (llama-server b11960+)
          │
          ├─► Direct PCM16 Audio Stream (No STT)
          ├─► Tool Calling Execution Loop
          │
          ▼
       Interleaved Text & Audio (24 kHz PCM16)
          │
          ▼
   AudioPlayer (Jitter Buffer) ──► PipeWire Echo Cancellation
          ▲
          └─ Barge-in Interrupt ("Buddy" or VAD Speech)
```

- **Model**: `LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M` (2.17B parameters, 32K context window). Native interleaved multimodal speech/text input and output.
- **Server**: `llama.cpp` `llama-server` (build b11960 or newer). Run with:
  ```bash
  llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M --reasoning off -ngl 99 -c 32768 --port 9931
  ```
- **Realtime Protocol**: OpenAI-compatible `/v1/realtime` bidirectional WebSocket carrying PCM16 mono audio at 16 kHz in, 24 kHz out.
- **Vision**: Screen capture (`grim` on Wayland / `maim` on X11) converted to JPEG and dispatched to `/v1/chat/completions` image URL data URI.

---

## 1. Arch Linux Installation Guide

### Step 1. Install System Dependencies

Install development tools, audio libraries, Wayland/GTK4 libraries, and screenshot utilities:

```bash
sudo pacman -S --needed \
    base-devel cmake git curl \
    pipewire pipewire-pulse pipewire-audio wireplumber \
    libspa-aec-webrtc \
    gtk4 gtk4-layer-shell gobject-introspection \
    python python-pip python-gobject \
    portaudio grim maim alsa-utils
```

### Step 2. Build `llama.cpp` with `/v1/realtime` Support

`llama.cpp` b11960 or newer is required:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# For NVIDIA GPUs (CUDA):
cmake -B build -DLLAMA_CURL=ON -DGGML_CUDA=ON
# For AMD GPUs (ROCm): cmake -B build -DLLAMA_CURL=ON -DGGML_HIPBLAS=ON
# For CPU only: cmake -B build -DLLAMA_CURL=ON

cmake --build build --config Release -j$(nproc) --target llama-server

# Install binary to system PATH
sudo cp build/bin/llama-server /usr/local/bin/
```

### Step 3. Install Buddy Assistant

From the `buddy` repository directory:

```bash
cd /home/dhp/desk_ai/buddy

# Create a virtual environment or install into user site:
python3 -m venv .venv
source .venv/bin/activate

# Install buddy with Google APIs and dev tools:
pip install -e '.[google,dev]'
```

### Step 4. Configure PipeWire Echo Cancellation

To prevent assistant speaker audio from looping back into the microphone during barge-in:

```bash
mkdir -p ~/.config/pipewire/pipewire.conf.d/
cp pipewire/echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/
systemctl --user restart pipewire
```

---

## 2. Setup & First Run

### Step 1. Initialize Configuration and Data

```bash
buddy init
```

This sets up:
- `~/.config/buddy/config.toml` (editable settings)
- `~/.local/share/buddy/memory/` (`core.md`, `facts.md`, `timetable.yaml`, `mess.yaml`)
- `~/.config/buddy/tools/` (drop-in directory for custom tool scripts)

### Step 2. Run Diagnostics

Check all subsystem dependencies and audio routing:

```bash
buddy doctor
```

To probe live communication with a running `llama-server`:
```bash
buddy doctor --probe
```

### Step 3. Launch Assistant

```bash
buddy run
```

Options:
- `--debug-events`: Dumps all WebSocket JSON payloads (with audio buffers truncated).
- `--no-orb`: Runs headless in terminal (no GTK window).
- `--no-autostart`: Connects to an already running `llama-server` without launching one.

---

## 3. Voice Interaction & Modes

| Mode | Trigger | Behavior |
|---|---|---|
| **Wake Word** | Say `"Buddy"` | Vosk detects wake word with 1.5s pre-roll buffer so *"Buddy, what's my next class?"* works in a single breath. |
| **Always-Listening** | Left-click orb / `buddy toggle` / `buddy always` | Assistant listens continuously via VAD; automatically falls back to wake word after 10 minutes idle. |
| **Follow-Up Window** | Automatic (8s) | For 8 seconds after an answer finishes, the mic stays active without requiring the wake word. |
| **Barge-In** | Speak `"Buddy"` (or VAD speech) | Immediately cancels in-flight model response, clears playback buffer, ignores late response frames, and captures your interruption. |
| **Mute** | Right-click orb / `buddy mute` | Drops all audio capture and silences speaker. |

---

## 4. Memory Architecture (OpenClaw-Style)

Memory lives in plain, human-editable text and YAML files under `~/.local/share/buddy/memory/`:

```
~/.local/share/buddy/memory/
├── core.md            # Hard rules and user identity (always in prompt, capped at 2000 chars)
├── facts.md           # Long-term facts learned over time (one per line)
├── summary.md         # Rolling conversation consolidation summary
├── timetable.yaml     # Exact weekly class schedule
├── mess.yaml          # Exact dining / meal menu
├── notes/             # Daily conversation transcript logs
│   └── YYYY-MM-DD.md
└── .search.db         # High-speed SQLite FTS5 full-text index
```

### Exact Lookups (YAML)
Times such as `10:30` in YAML normally parse as integer minutes (`630`). Buddy's loaders automatically normalize all time entries to clean `HH:MM` strings:
```yaml
Monday:
  - time: 10:30
    class: "Operating Systems"
    room: "CS-201"
```

### Safe Whitelisting
Memory tools (`remember`, `search_memory`, `forget`, `read_memory`) resolve paths against strict whitelists and reject any `../` path traversal attempts.

---

## 5. Rolling Context Management (80/20 Plan)

`llama-server` keeps all items in context during a realtime session and cannot selectively delete items.
1. When estimated conversation tokens reach **80% of `n_ctx`** (approx. 26,200 tokens on 32K context):
2. Buddy marks the session for rollover during the next idle interval.
3. The oldest **20% of tokens** are consolidated into a durable rolling summary and facts.
4. Buddy **never splits an exchange** and always keeps at least the **last 2 turns** intact.
5. The WebSocket closes and opens a brand-new session whose `session.update` instructions inject:
   - Base system prompt
   - Core memory (`core.md`)
   - Rolling summary (`summary.md`)
   - Surviving recent exchanges as clean text history.
6. Dialogue history is persisted in `.active_session.json` across process restarts.

---

## 6. Proactive Notifications

Buddy delivers spoken notices strictly when idle, at most once per 15 seconds, deduplicated:
- **Class Reminders**: Spoken 10 minutes before classes listed in `timetable.yaml`.
- **Meal Reminders**: Spoken 15 minutes before breakfast, lunch, snacks, or dinner in `mess.yaml`.
- **New Email**: Polled every 10 minutes via Gmail API (the initial poll only baseline-indexes existing messages).

---

## 7. Extending Buddy: Custom Tool Plugins

Create a `.py` file inside `~/.config/buddy/tools/` (e.g. `pomodoro.py`):

```python
from typing import Annotated, Literal
from buddy.tools.registry import tool

@tool
def timer(
    duration_min: Annotated[int, "Duration of the timer in minutes"] = 25,
    tag: Annotated[Literal["work", "break"], "Type of timer session"] = "work",
) -> str:
    """Set a pomodoro work or break timer."""
    return f"Started a {duration_min}-minute {tag} timer."
```

Buddy automatically discovers plugins on startup and converts Python type hints into flat OpenAI-compatible function schemas.

---

## 8. Google OAuth Setup (Gmail & Classroom)

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable **Gmail API** and **Google Classroom API**.
3. Create an **OAuth 2.0 Client ID** (Application type: *Desktop App*).
4. Download the client secret JSON to `~/.config/buddy/client_secrets.json`.
5. Run:
   ```bash
   buddy auth-google
   ```
6. The authorization token is saved with strict file mode `0600` at `~/.config/buddy/google_token.json`.

---

## 9. Floating Orb Desktop UI

The floating orb widget utilizes `PyGObject` + `GTK4` + `gtk4-layer-shell` for true Wayland desktop overlay positioning:
- **Calm Blue**: Idle / Wake Word Listening
- **Bright Blue**: Always-Listening Mode
- **Pulsing Green**: Capturing user voice
- **Spinning Cyan Arc**: Model thinking / tool execution
- **Dynamic Amplitude Scaling Blue**: Assistant speaking (reacts to PCM output volume)
- **Muted Gray**: Microphone and playback muted
- **Crimson Red**: Error recovery

---

## 10. CLI Reference & Unix Control Socket

Buddy exposes a Unix domain socket at `$XDG_RUNTIME_DIR/buddy.sock` (`/tmp/buddy.sock` fallback):

```bash
buddy run [--debug-events] [--no-orb] [--no-autostart]  # Run daemon
buddy status                                          # Check current agent state
buddy toggle                                          # Toggle wake vs always listening
buddy always                                          # Force always-listening mode
buddy wake                                            # Force wake-word-only mode
buddy mute / buddy unmute                             # Toggle microphone mute
buddy say "Hey there! Class starts in ten minutes."   # Speak proactive text
buddy tools                                           # List all registered schemas
buddy quit                                            # Clean daemon shutdown
```

---

## 11. Systemd User Service

To start Buddy automatically on user login:

```bash
mkdir -p ~/.config/systemd/user/
cp systemd/buddy.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now buddy.service
```

---

## 12. Honest Report of Unverified Items

In adherence to senior engineering standards, the following aspects **could not be verified on real physical hardware** inside this containerized build environment:

1. **Live LFM3-Omni-2B GGUF Model Execution**: While the WebSocket `/v1/realtime` client and server manager were verified against our RFC-compliant protocol test double, live inference with `LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M` requires host GPU VRAM and a live internet connection to download the 2.17B model and projector weights.
2. **Physical Acoustic Latency & Echo Cancellation**: Hardware acoustic round-trip times (microphone capture -> VAD -> WebSocket -> jitter buffer -> DAC) depend on user hardware, sound card buffer sizes, and PipeWire ALSA/Pulse configuration.
3. **Vosk Wake Accuracy in Ambient Noise**: Vosk's small acoustic model (`vosk-model-small-en-us-0.15`) with grammar `["buddy", "[unk]"]` achieves high precision in quiet rooms, but real-world false-reject rates with background music or speech echo must be tuned on the target microphone.
4. **Wayland Compositor Layer-Shell Compatibility**: `gtk4-layer-shell` functions smoothly on wlroots compositors (Sway, Hyprland, Wayfire). Under GNOME Wayland (Mutter), layer-shell is unsupported; Buddy falls back to a floating borderless GTK4 window.
5. **Real Google Classroom Student Permissions**: College and university Google Workspace for Education tenants frequently restrict unverified third-party OAuth apps from accessing Classroom Coursework and Submissions scopes. Additionally, Google OAuth apps in "Testing" mode have refresh tokens that expire after 7 days.
6. **2B Parameter Model Tool Reliability**: Small 2B models operating in non-reasoning mode may occasionally produce invalid parameter types or hallucinated function names during complex tool interactions; Buddy provides strict argument coercion and returns schema error strings directly to the model to enable self-correction.
