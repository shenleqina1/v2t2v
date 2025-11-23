# V2T2V 实时语音转写与语音合成套件

V2T2V（Voice-to-Text-to-Voice）是一套面向 Windows 平台的语音工具，包含一个可部署在本地或局域网的流式语音识别服务器（`server.py`）以及一个功能完整的图形客户端（`vtt.py`）。客户端既能离线使用 Vosk 模型，也能连接远程 Whisper 服务器，还内置多种 TTS 引擎、热键、推按说话（PTT）和紧急急停流程。

---

## 功能概览

| 组件 | 主要能力 | 说明 |
| --- | --- | --- |
| 语音识别服务器 (`server.py`) | TCP 流式识别、Whisper/faster-whisper、VAD、声纹过滤 | 支持多客户端，默认 16 kHz 单声道 PCM；可开启 VAD 与声纹校验，只处理目标说话人 |
| 图形客户端 (`vtt.py`) | 本地/远程识别、实时字幕、TTS、热键、悬浮窗 | Windows 原生 UI，支持 Vosk 离线识别、远程服务器测试、推按说话、紧急停止、朗读缓存 |
| TTS 管理 | 微软 SAPI、Edge TTS、gTTS、pyttsx3、多音色/音量/语速 | 可为朗读配置输出设备、语种、语速音量，支持播放热键与紧急停止 |
| 快捷键体系 | 录音控制、PTT、急停、自动热键播放 | 支持鼠标/键盘组合，自带热键捕获器与 overlay 提示 |

---

## 目录结构

```
.
├── server.py              # TCP 语音识别服务器
├── vtt.py                 # GUI 客户端
├── speaker_verifier.py    # 声纹比对工具
├── model-ct2/             # faster-whisper (ctranslate2) 模型目录，可替换
├── vosk-model-cn-0.22/    # Vosk 中文模型（示例）
├── vosk-model-small-cn-0.22/
├── logs/                  # 运行日志
├── vtt-settings.json      # 客户端保存的 UI 与热键配置
└── requirements.txt       # Python 依赖
```

---

## 环境要求

- Python 3.9 及以上版本（建议 64 位）。
- Windows 10/11 推荐（客户端涉及 pywin32、SAPI、SendInput）；服务器可在 Windows/Linux 运行。
- GPU（可选）：若在服务器端使用大型 Whisper 模型，建议 NVIDIA GPU + CUDA 11.8+。
- FFmpeg（可选）：用于 gTTS/Edge TTS 合成后的音频转换。

---

## 快速开始

1. **创建虚拟环境并安装依赖**
   ```powershell
  cd \v2t2v-main
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   python vtt.py
   ```

2. **准备模型**（见下一节）。
3. **按需启动服务器和/或客户端**。

---

## 模型准备

### Whisper / faster-whisper（服务器）
- 默认从 `model-ct2/` 读取 ctranslate2 加速模型，若不存在则回退到 `model/`。
- 可从 [Hugging Face](https://huggingface.co/Systran) 下载 `faster-whisper-small/medium/large-v2` 等模型，解压后放到 `model-ct2/`。
- 若需原版 OpenAI Whisper，放入 `model/` 并通过启动参数 `--model` 指定路径。
  - 推荐下载（大小为解压后近似值，显存为 GPU 推理参考）：

    | 模型 | 大小 | 性能特点 | 计算资源 |
    | --- | --- | --- | --- |
    | [faster-whisper-large-v2](https://huggingface.co/Systran/faster-whisper-large-v2) | ≈ 11 GB | 最佳精度，流式延迟略高 | 需要 ≥12 GB VRAM；CPU 模式需高性能多核 |
    | [faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) | ≈ 5.5 GB | 精度与速度平衡 | 建议 8 GB VRAM；CPU 推理需 AVX2、线程数 ≥8 |
    | [faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | ≈ 2.9 GB | 适合中端显卡或实时需求 | 4–6 GB VRAM 可流畅；CPU 模式实时性一般 |
    | [faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base) | ≈ 1.6 GB | 轻量快速，精度略低 | 4 GB VRAM 或纯 CPU 均可运行 |

### Vosk（客户端本地识别）
- 将中文模型解压到 `vosk-model-cn-0.22/` 或 `vosk-model-small-cn-0.22/`。
- 在客户端 UI 中指定模型目录后即可离线识别。
  - 推荐下载（大小为解压后近似值）：

    | 模型 | 大小 | 性能特点 | 占用建议 |
    | --- | --- | --- | --- |
    | [vosk-model-cn-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip) | ≈ 3.8 GB | 全量中文词汇，高精度 | 运行时常驻内存约 1.5 GB，适合桌面 CPU |
    | [vosk-model-small-cn-0.22.zip](https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip) | ≈ 50 MB（压缩包），解压后 ≈ 210 MB | 轻量，识别速度快但词汇有限 | 仅占用数百 MB 内存，适合低配设备 |

---

## 运行服务器

```powershell
python server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --model model-ct2 \
  --device cuda \
  --compute-type int8_float16 \
  --speaker-ref speaker.wav \
  --speaker-threshold 0.8
```

常用参数：
- `--speaker-ref`: 可选，提供 16 kHz WAV 作为目标说话人；未命中将拒绝结果。
- `--vad-disable`: 关闭内置 VAD；默认启用（需 `faster-whisper.vad`）。
- `--device`: `cuda`/`cpu`，取决于硬件。
- `--compute-type`: `float16`、`int8_float16` 等，影响显存/速度。

服务器日志写入控制台与 `logs/whisper_server.log`（若配置）。

---

## 运行客户端

```powershell
python vtt.py
```

客户端要点：
1. **推理模式**：在“推理模式”下拉框选择“本地推理”（Vosk）或“服务器推理”（Whisper）。
2. **服务器参数**：切换到服务器推理后，填写 IP/端口，点击“测试服务器连通性”验证。测试音频会缓存在 `remote-test-sample.wav`，避免重复合成。
3. **TTS 引擎与音色**：在“语音合成”区域选择引擎、输出设备、语种/音色，可调整语速/音量。
4. **字幕/朗读**：勾选“自动朗读”即可由当前 TTS 引擎播放新的识别结果。
5. **悬浮窗/Overlay**：可启用桌面浮窗显示当前识别状态，并可拖拽定位。

---

## 快捷键、PTT 与紧急急停

- **常规热键**：在“快捷键”栏点击“捕获”，按下组合键/鼠标键即可注册；可用于触发朗读、清屏等。
- **推按说话 (PTT)**：启用后，只有在指定组合被按住时才会采集音频。支持键鼠混合。
- **紧急急停**：
  1. 在“急停快捷键”一栏捕获组合键。
  2. 一旦触发，`TTSManager.emergency_stop()` 会立即：
     - 停止当前识别会话或录音。
     - 清空 TTS 播放队列并调用当前引擎的 `stop()`。
     - 通过状态栏提示用户。
  3. UI 上的“紧急停止”按钮与热键共享同一逻辑，可在任何状态下强制终止朗读。

---

## 日志与设置

- **客户端日志**：`logs/vtt-client.log`（自动轮转）。
- **服务器日志**：默认输出到 stdout，可按需配置。
- **用户设置**：`vtt-settings.json` 保存推理模式、热键、语速、音量等，可删除以重置。

---

## 常见问题

1. **“找不到声音设备/缺少 sounddevice”**：确保已安装 `sounddevice` 并允许麦克风权限。ASIO 驱动请切换到可被 Windows 共享的设备。
2. **“没有检测到 pywin32 / pythoncom”**：客户端需在 Windows 上以管理员/普通用户安装 `pywin32`，`pip install pywin32` 后运行 `python -m pywin32_postinstall install`（若自动脚本未执行）。
3. **GPU 显存不足**：改用 `--compute-type int8_float16` 或选择更小的模型（如 `small`）。
4. **TTS 无法播放或急停无效**：检查是否选中可用的 TTS 引擎；Edge/gTTS 需要有效网络与 FFmpeg；急停热键需保持监听器运行（界面会提示状态）。
5. **服务器连通性测试 CPU 飙升**：测试音频仅在首次生成时消耗 TTS 资源，之后复用缓存；若仍占用过高，可手动删除 `remote-test-sample.wav` 并缩短测试文本。

---

## 开发与调试

- 建议使用 VS Code 或 PyCharm，并配置 `PYTHONPATH` 指向仓库根目录。
- 可在 `.env` 中追加常用参数（例如服务器地址），或编写自定义启动脚本。
- 运行单文件语法检查：
  ```powershell
  python -m py_compile vtt.py
  python -m py_compile server.py
  ```

欢迎根据实际需求扩展 UI、接入新的 TTS/ASR 引擎，或提交 Issue/PR 反馈问题。

