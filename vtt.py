from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import wave
import math
from io import BytesIO
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Set, Tuple

if sys.platform == "win32":  # pragma: no cover - platform specific
    import ctypes
    from ctypes import wintypes

    _ULONG_PTR = getattr(wintypes, "ULONG_PTR", None)
    if _ULONG_PTR is None:  # pragma: no cover - older Python
        _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    _USER32 = ctypes.WinDLL("user32", use_last_error=True)

    class _MOUSEINPUT(ctypes.Structure):  # pragma: no cover - struct definition
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _KEYBDINPUT(ctypes.Structure):  # pragma: no cover - struct definition
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _HARDWAREINPUT(ctypes.Structure):  # pragma: no cover - struct definition
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTUNION(ctypes.Union):  # pragma: no cover - struct definition
        _fields_ = [
            ("mi", _MOUSEINPUT),
            ("ki", _KEYBDINPUT),
            ("hi", _HARDWAREINPUT),
        ]

    class _INPUT(ctypes.Structure):  # pragma: no cover - struct definition
        _fields_ = [
            ("type", wintypes.DWORD),
            ("union", _INPUTUNION),
        ]

    _INPUT_TYPE_KEYBOARD = 1
    _KEYEVENTF_EXTENDEDKEY = 0x0001
    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_SCANCODE = 0x0008

    def _build_keyboard_input(vk_code: int, key_down: bool) -> _INPUT:
        scan_code = _USER32.MapVirtualKeyW(vk_code, 0)
        flags = _KEYEVENTF_SCANCODE
        w_vk = 0
        w_scan = scan_code & 0xFF
        if scan_code == 0:
            w_vk = vk_code
            w_scan = 0
            flags = 0
        else:
            if scan_code & 0x0100:
                flags |= _KEYEVENTF_EXTENDEDKEY
            w_scan = scan_code & 0xFF
        if not key_down:
            flags |= _KEYEVENTF_KEYUP
        return _INPUT(
            type=_INPUT_TYPE_KEYBOARD,
            union=_INPUTUNION(
                ki=_KEYBDINPUT(
                    wVk=w_vk,
                    wScan=w_scan,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )

    def _send_keyboard_inputs(inputs: List[object]) -> None:
        if not inputs:
            return
        array_type = _INPUT * len(inputs)
        buffer = array_type(*inputs)
        sent = _USER32.SendInput(len(buffer), buffer, ctypes.sizeof(_INPUT))
        if sent != len(buffer):  # pragma: no cover - defensive
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "SendInput failed")

else:  # pragma: no cover - non-Windows stub
    _USER32 = None

    def _build_keyboard_input(vk_code: int, key_down: bool) -> None:
        raise NotImplementedError

    def _send_keyboard_inputs(inputs: List[object]) -> None:
        raise NotImplementedError

# Logging setup
LOGGER = logging.getLogger("vtt.client")
_LOGGING_CONFIGURED = False


def _configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, "vtt-client.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    handlers: List[logging.Handler] = [stream_handler]
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    except OSError:
        file_handler = None
    if file_handler is not None:
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        for handler in handlers:
            root_logger.addHandler(handler)
    else:
        for handler in handlers:
            root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    _LOGGING_CONFIGURED = True


# Third-party dependencies. The UI will highlight missing pieces.
try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - guidance via UI
    sd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from vosk import KaldiRecognizer, Model  # type: ignore
except ImportError:  # pragma: no cover - guidance via UI
    KaldiRecognizer = None  # type: ignore
    Model = None  # type: ignore

try:  # pragma: no cover - Windows specific
    import win32com.client  # type: ignore
except ImportError:  # pragma: no cover - fallback to pyttsx3
    win32com = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyttsx3  # type: ignore
except ImportError:  # pragma: no cover - handled in GUI
    pyttsx3 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - handled in GUI
    pyautogui = None  # type: ignore
else:
    try:
        pyautogui.FAILSAFE = False
    except Exception:
        pass

try:  # pragma: no cover - optional dependency
    from pynput import keyboard as pynput_keyboard  # type: ignore
    from pynput import mouse as pynput_mouse  # type: ignore
except ImportError:  # pragma: no cover - handled in GUI
    pynput_keyboard = None  # type: ignore
    pynput_mouse = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import edge_tts  # type: ignore
except ImportError:  # pragma: no cover - handled in GUI
    edge_tts = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from gtts import gTTS  # type: ignore
    from gtts.lang import tts_langs  # type: ignore
except ImportError:  # pragma: no cover - handled in GUI
    gTTS = None  # type: ignore
    tts_langs = None  # type: ignore


def _find_ffmpeg() -> Optional[str]:
    executable = shutil.which("ffmpeg")
    if executable is None and sys.platform == "win32":
        executable = shutil.which("ffmpeg.exe")
    return executable


def _convert_mp3_to_wav(
    mp3_bytes: bytes,
    ffmpeg_executable: str,
    *,
    speed: float = 1.0,
    volume: float = 1.0,
) -> bytes:
    mp3_path = None
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
            mp3_file.write(mp3_bytes)
            mp3_path = mp3_file.name
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        filters: List[str] = []
        tempo = max(0.5, min(2.0, float(speed)))
        if not math.isclose(tempo, 1.0, rel_tol=1e-3):
            filters.append(f"atempo={tempo:.3f}")
        gain = max(0.0, min(2.5, float(volume)))
        if not math.isclose(gain, 1.0, rel_tol=1e-3):
            filters.append(f"volume={gain:.3f}")
        command = [ffmpeg_executable, "-y", "-i", mp3_path]
        if filters:
            command.extend(["-filter:a", ",".join(filters)])
        command.extend(["-ar", "24000", "-ac", "1", wav_path])
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(detail or "ffmpeg 转换失败")
        with open(wav_path, "rb") as wav_file:
            wav_data = wav_file.read()
        # 用完立即删除
        try:
            if mp3_path and os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass
        return wav_data
    finally:
        # 再次兜底删除
        try:
            if mp3_path and os.path.exists(mp3_path):
                os.remove(mp3_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


def _play_wav_bytes(data: bytes, stop_event: Optional[threading.Event] = None) -> None:
    if not data:
        return
    if sys.platform != "win32":
        raise RuntimeError("当前系统不支持播放 WAV 音频。")
    import winsound

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name
    try:
        if stop_event is not None and stop_event.is_set():
            winsound.PlaySound(None, getattr(winsound, "SND_PURGE", 0))
            return

        duration: Optional[float] = None
        try:
            with wave.open(temp_path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate() or 24000
                if rate > 0:
                    duration = frames / float(rate)
        except wave.Error:
            duration = None

        flags = getattr(winsound, "SND_FILENAME", 0x00020000) | getattr(winsound, "SND_ASYNC", 0x0001)
        winsound.PlaySound(temp_path, flags)

        if duration is None:
            # 估算播放时长（假设 24kHz 单声道 16bit）。
            duration = max(len(data) / (24000 * 2), 0.1)

        deadline = time.monotonic() + max(duration, 0.0)
        interval = 0.05
        while True:
            if stop_event is not None and stop_event.is_set():
                winsound.PlaySound(None, getattr(winsound, "SND_PURGE", 0))
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(interval)

        winsound.PlaySound(None, getattr(winsound, "SND_PURGE", 0))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


class RecognitionController:
    """Streams microphone audio into Vosk and reports results."""

    def __init__(self, app: "TranscriptionApp") -> None:
        self.app = app
        self._audio_queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self._recognizer: Optional[object] = None
        self._running = threading.Event()
        self._model: Optional[object] = None
        self._initializer: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._state_lock = threading.Lock()
        self._preloaded_model: Optional[object] = None
        self._preloaded_model_path: Optional[str] = None
        self._preloaded_sample_rate: Optional[int] = None
        self._preload_thread: Optional[threading.Thread] = None
        self._requested_preload_path: Optional[str] = None
        self._requested_preload_rate: Optional[int] = None

    def start(self, device_index: Optional[int], model_path: str, sample_rate: int = 16000) -> None:
        if sd is None:
            raise RuntimeError("缺少 sounddevice 库，请执行 `pip install sounddevice`。")
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("缺少 vosk 库，请执行 `pip install vosk`。")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"未找到模型目录: {model_path}")

        LOGGER.info(
            "[local] starting recognition device=%s model=%s sample_rate=%s",
            device_index,
            model_path,
            sample_rate,
        )

        with self._state_lock:
            if self._initializer and self._initializer.is_alive():
                raise RuntimeError("识别正在初始化，请稍候...")

        self.stop()
        stop_event = threading.Event()
        self._running.clear()
        with self._state_lock:
            self._stop_requested = stop_event

        initializer = threading.Thread(
            target=self._initialize_and_start,
            args=(device_index, model_path, sample_rate, stop_event),
            name="VoskInit",
            daemon=True,
        )
        with self._state_lock:
            self._initializer = initializer
        initializer.start()

    def _initialize_and_start(
        self,
        device_index: Optional[int],
        model_path: str,
        sample_rate: int,
        stop_event: threading.Event,
    ) -> None:
        stream: Optional[object] = None
        worker: Optional[threading.Thread] = None
        try:
            try:
                with self._state_lock:
                    cached_model = None
                    if self._preloaded_model_path == model_path and self._preloaded_model is not None:
                        cached_model = self._preloaded_model
                if cached_model is not None:
                    model = cached_model
                recognizer = KaldiRecognizer(model, sample_rate)
                recognizer.SetWords(True)
                with self._state_lock:
                    if self._preloaded_model is None or self._preloaded_model_path != model_path:
                        self._preloaded_model = model
                        self._preloaded_model_path = model_path
                        self._preloaded_sample_rate = sample_rate
            except Exception as exc:  # pragma: no cover - defensive
                self._handle_initialization_failure(exc, stop_event)
                return

            if stop_event.is_set():
                return

            with self._state_lock:
                self._model = model
                self._recognizer = recognizer
                self._audio_queue = queue.Queue()

            def audio_callback(indata, frames, time_info, status):  # type: ignore[override]
                if stop_event.is_set() or not self._running.is_set():
                    return
                if status:
                    self.app.post_status(f"音频流状态: {status}")
                self._audio_queue.put(bytes(indata))

            try:
                stream = sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=8000,
                    device=device_index,
                    dtype="int16",
                    channels=1,
                    callback=audio_callback,
                )
            except Exception as exc:  # pragma: no cover - device setup
                with self._state_lock:
                    self._model = None
                    self._recognizer = None
                self._handle_initialization_failure(exc, stop_event)
                return

            with self._state_lock:
                self._stream = stream

            if stop_event.is_set():
                self._close_stream(stream)
                with self._state_lock:
                    if self._stream is stream:
                        self._stream = None
                    self._model = None
                    self._recognizer = None
                return

            self._running.set()
            try:
                stream.start()
            except Exception as exc:  # pragma: no cover - device start
                self._running.clear()
                self._close_stream(stream)
                with self._state_lock:
                    if self._stream is stream:
                        self._stream = None
                    self._model = None
                    self._recognizer = None
                self._handle_initialization_failure(exc, stop_event)
                return

            if stop_event.is_set():
                self._running.clear()
                self._close_stream(stream)
                with self._state_lock:
                    if self._stream is stream:
                        self._stream = None
                    self._model = None
                    self._recognizer = None
                return

            worker = threading.Thread(target=self._process_audio, name="VoskWorker", daemon=True)
            with self._state_lock:
                self._worker = worker
            worker.start()

            if stop_event.is_set():
                self._running.clear()
                self._close_stream(stream)
                if worker.is_alive():
                    worker.join(timeout=1)
                with self._state_lock:
                    if self._worker is worker:
                        self._worker = None
                    if self._stream is stream:
                        self._stream = None
                    self._model = None
                    self._recognizer = None
                return

            self.app.notify_recognition_ready()
        finally:
            self._finalize_initializer()

    def preload_model(self, model_path: str, sample_rate: int = 16000) -> None:
        if Model is None or KaldiRecognizer is None:
            return
        normalized_path = os.path.abspath(model_path)
        if not os.path.isdir(normalized_path):
            return
        with self._state_lock:
            if (
                self._preloaded_model is not None
                and self._preloaded_model_path == normalized_path
                and self._preloaded_sample_rate == sample_rate
            ):
                return
            self._requested_preload_path = normalized_path
            self._requested_preload_rate = sample_rate
            preload_thread = self._preload_thread
            if preload_thread is not None and preload_thread.is_alive():
                # 允许现有线程继续，但最终将检查请求是否仍匹配。
                pass
            thread = threading.Thread(
                target=self._preload_worker,
                args=(normalized_path, sample_rate),
                name="VoskPreload",
                daemon=True,
            )
            self._preload_thread = thread
        self.app.post_status("正在后台加载本地模型...")
        thread.start()

    def _preload_worker(self, model_path: str, sample_rate: int) -> None:
        try:
            model = Model(model_path)
        except Exception as exc:  # pragma: no cover - defensive
            if self._requested_preload_path == model_path:
                self.app.post_status(f"模型预加载失败: {exc}")
            with self._state_lock:
                if self._preload_thread is threading.current_thread():
                    self._preload_thread = None
            return

        with self._state_lock:
            if (
                self._requested_preload_path != model_path
                or self._requested_preload_rate != sample_rate
            ):
                # 新的预加载请求已覆盖当前目标，保留结果但不更新状态。
                if self._preload_thread is threading.current_thread():
                    self._preload_thread = None
                return
            self._preloaded_model = model
            self._preloaded_model_path = model_path
            self._preloaded_sample_rate = sample_rate
            if self._preload_thread is threading.current_thread():
                self._preload_thread = None
        self.app.post_status("本地模型预加载完成。")

    def _finalize_initializer(self) -> None:
        with self._state_lock:
            if self._initializer is threading.current_thread():
                self._initializer = None

    def _close_stream(self, stream: Optional[object]) -> None:
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            stream.close()
        except Exception:  # pragma: no cover - best effort
            pass

    def _handle_initialization_failure(self, exc: Exception, stop_event: threading.Event) -> None:
        if stop_event.is_set():
            return
        self.app.notify_recognition_failed(exc)

    def _process_audio(self) -> None:
        recognizer = self._recognizer
        if recognizer is None:
            return
        while self._running.is_set():
            try:
                data = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        LOGGER.info("[local] final: %s", text)
                        self.app.handle_final_result(text)
                else:
                    result = json.loads(recognizer.PartialResult())
                    partial = result.get("partial", "").strip()
                    if partial:
                        LOGGER.info("[local] partial: %s", partial)
                    self.app.handle_partial_result(partial)
            except Exception as exc:  # pragma: no cover - defensive
                self.app.post_status(f"识别出错: {exc}")

        try:
            final_json = json.loads(recognizer.FinalResult())
            text = final_json.get("text", "").strip()
            if text:
                LOGGER.info("[local] final: %s", text)
                self.app.handle_final_result(text)
        except Exception as exc:  # pragma: no cover - defensive
            self.app.post_status(f"识别结束处理失败: {exc}")

        self.app.handle_partial_result("")

    def stop(self) -> None:
        LOGGER.info("[local] stopping recognition")
        with self._state_lock:
            stop_event = self._stop_requested
            stream = self._stream
            worker = self._worker
            initializer = self._initializer
            self._stop_requested = threading.Event()

        self._running.clear()
        if stop_event is not None:
            stop_event.set()
        try:
            self._audio_queue.put_nowait(b"")
        except Exception:
            pass

        self._close_stream(stream)

        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1)

        if initializer is not None and initializer is not threading.current_thread():
            initializer.join(timeout=1)

        with self._state_lock:
            if self._worker is worker:
                self._worker = None
            if self._initializer is initializer:
                self._initializer = None
            if self._stream is stream:
                self._stream = None
            self._model = None
            self._recognizer = None

        self._audio_queue = queue.Queue()


class RemoteRecognitionController:
    """Streams microphone audio to a remote recognition server."""

    def __init__(self, app: "TranscriptionApp") -> None:
        self.app = app
        self._audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._stream: Optional[object] = None
        self._running = threading.Event()
        self._stopping = False
        self._socket: Optional[socket.socket] = None
        self._writer = None
        self._reader = None
        self._sender: Optional[threading.Thread] = None
        self._receiver: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._sent_audio_chunks = 0
        self._last_partial_text = ""
        self._final_received = False

    def start(self, device_index: Optional[int], host: str, port: int, sample_rate: int = 16000) -> None:
        if sd is None:
            raise RuntimeError("缺少 sounddevice 库，请执行 `pip install sounddevice`。")
        if not host:
            raise ValueError("服务器地址不能为空。")
        if port <= 0 or port > 65535:
            raise ValueError("请输入有效的端口号。")

        self.stop()
        self._running.clear()
        self._audio_queue = queue.Queue()
        self._sent_audio_chunks = 0
        self._last_partial_text = ""
        self._final_received = False

        LOGGER.info(
            "[remote] connecting to %s:%s device=%s sample_rate=%s",
            host,
            port,
            device_index,
            sample_rate,
        )

        try:
            sock = socket.create_connection((host, port), timeout=5)
        except OSError as exc:
            raise RuntimeError(f"无法连接服务器: {exc}") from exc

        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(5.0)
        self._socket = sock
        self._writer = sock.makefile("wb")
        self._reader = sock.makefile("rb")

        try:
            self._send_json({"type": "hello", "sample_rate": sample_rate, "client": "vtt"})
            ack_raw = self._reader.readline()
            if not ack_raw:
                raise RuntimeError("服务器未响应握手。")
            ack = json.loads(ack_raw.decode("utf-8"))
            if ack.get("status") != "ok":
                message = ack.get("message", "服务器拒绝连接。")
                raise RuntimeError(message)
        except Exception:
            self._cleanup_connection()
            raise
        finally:
            if self._socket is not None:
                try:
                    self._socket.settimeout(None)
                except OSError:
                    pass

        def audio_callback(indata, frames, time_info, status):  # type: ignore[override]
            if not self._running.is_set():
                return
            if status:
                self.app.post_status(f"音频流状态: {status}")
            self._audio_queue.put(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=8000,
            device=device_index,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        )

        try:
            self._stream.start()
        except Exception:
            self._cleanup_connection()
            self._stream = None
            raise

        self._running.set()
        self._stopping = False
        self._sender = threading.Thread(target=self._sender_loop, name="RemoteSender", daemon=True)
        self._receiver = threading.Thread(target=self._receiver_loop, name="RemoteReceiver", daemon=True)
        self._sender.start()
        self._receiver.start()
        self.app.post_status("服务器连接成功，开始识别。")
        self.app.notify_recognition_ready()
        LOGGER.info("[remote] connection established")

    def stop(self) -> None:
        running = self._running.is_set()
        self._stopping = True
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            finally:
                self._stream = None
        if running:
            try:
                self._send_json({"type": "bye"})
            except Exception:
                pass
        self._audio_queue.put(None)
        sender_thread = self._sender
        receiver_thread = self._receiver
        if sender_thread is not None and sender_thread is not threading.current_thread():
            sender_thread.join(timeout=1)
            if sender_thread.is_alive():
                LOGGER.info("[remote] sender still flushing audio after timeout")
        if receiver_thread is not None and receiver_thread is not threading.current_thread():
            wait_timeout = 1.0
            receiver_thread.join(timeout=wait_timeout)
            if receiver_thread.is_alive() and self._stopping and not self._final_received:
                extra_wait = 5.0
                LOGGER.info("[remote] waiting up to %.1fs for final result", extra_wait)
                receiver_thread.join(timeout=extra_wait)
            if receiver_thread.is_alive():
                LOGGER.warning("[remote] receiver thread未及时退出，将继续关闭连接")
        self._sender = None
        self._receiver = None
        self._running.clear()
        fallback_text = ""
        if self._stopping and not self._final_received:
            fallback_text = self._last_partial_text.strip()
        self._last_partial_text = ""
        self._final_received = False
        self._cleanup_connection()
        if fallback_text:
            LOGGER.info("[remote] synthesizing final from partial: %s", fallback_text)
            self.app.handle_final_result(fallback_text)
        self._stopping = False
        LOGGER.info("[remote] connection closed")

    def _sender_loop(self) -> None:
        while True:
            try:
                chunk = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if chunk is None:
                break
            try:
                payload = base64.b64encode(chunk).decode("ascii")
                self._sent_audio_chunks += 1
                if self._sent_audio_chunks <= 3:
                    LOGGER.info(
                        "[remote] sending chunk #%s size=%s",
                        self._sent_audio_chunks,
                        len(chunk),
                    )
                self._send_json({"type": "audio", "data": payload})
            except Exception as exc:
                self._handle_connection_error(f"发送音频失败: {exc}")
                break

    def _receiver_loop(self) -> None:
        while self._running.is_set():
            if self._reader is None:
                break
            try:
                raw = self._reader.readline()
            except Exception as exc:
                if self._stopping:
                    LOGGER.info("[remote] receiver exiting during stop: %s", exc)
                    break
                self._handle_connection_error(f"连接读取失败: {exc}")
                break
            if not raw:
                if self._stopping:
                    LOGGER.info("[remote] receiver closed after stop request")
                    break
                LOGGER.error("[remote] connection error: 服务器断开连接。")
                self._handle_connection_error("服务器断开连接。")
                break
            try:
                message = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            msg_type = message.get("type")
            if msg_type == "partial":
                partial = message.get("text", "")
                if partial:
                    self._last_partial_text = partial
                if partial:
                    LOGGER.info("[remote] partial: %s", partial)
                self.app.handle_partial_result(partial)
            elif msg_type == "final":
                text = message.get("text", "")
                if text:
                    LOGGER.info("[remote] final: %s", text)
                    self.app.handle_final_result(text)
                    self._final_received = True
                    self._last_partial_text = ""
            elif msg_type == "error":
                error_text = message.get("message", "服务器处理出错。")
                self._handle_connection_error(error_text)
                break

    def _handle_connection_error(self, message: str) -> None:
        if self._stopping:
            LOGGER.info("[remote] connection closed: %s", message)
        else:
            LOGGER.error("[remote] connection error: %s", message)
            if self._running.is_set():
                self.app.notify_recognition_failed(RuntimeError(message))
        self._running.clear()
        self._cleanup_connection()
        self.app.handle_partial_result("")

    def _send_json(self, payload: Dict[str, object]) -> None:
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._writer_lock:
            if self._writer is None:
                raise RuntimeError("连接已关闭。")
            self._writer.write(data)
            self._writer.flush()

    def _cleanup_connection(self) -> None:
        with self._writer_lock:
            writer = self._writer
            reader = self._reader
            sock = self._socket
            self._writer = None
            self._reader = None
            self._socket = None
        for fp in (writer, reader):
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


class BaseTTS:
    """Shared interface for speech synthesis back-ends."""

    def list_outputs(self) -> List[Tuple[str, Optional[str]]]:
        return [("默认输出设备", None)]

    def set_output(self, token_id: Optional[str]) -> None:
        raise NotImplementedError

    def list_voices(self) -> List[VoiceOption]:
        return []

    def set_voice(self, token_id: Optional[str]) -> None:
        raise NotImplementedError

    def set_rate(self, rate: float) -> bool:
        return False

    def set_volume(self, volume: float) -> bool:
        return False

    def speak(self, text: str) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def attach_stop_event(self, stop_event: threading.Event) -> None:
        pass


class SapiTTS(BaseTTS):
    def __init__(self) -> None:
        if win32com is None:
            raise RuntimeError("需要安装 pywin32 才能使用 Windows SAPI 输出。")
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._lock = threading.Lock()
        self._outputs: List[Tuple[str, Optional[str]]] = []
        self._outputs = self._collect_outputs()
        self._voices = self._collect_voices()
        self._stop_event: Optional[threading.Event] = None
        try:
            current_output = self._voice.AudioOutput
            self._current_output_id: Optional[str] = getattr(current_output, "Id", None)
        except Exception:
            self._current_output_id = None

    def _collect_outputs(self) -> List[Tuple[str, Optional[str]]] :
        import pythoncom

        outputs: List[Tuple[str, Optional[str]]] = []
        pythoncom.CoInitialize()
        try:
            try:
                collection = self._voice.GetAudioOutputs()
            except Exception:
                collection = None
            if collection:
                for idx in range(collection.Count):
                    token = collection.Item(idx)
                    token_id = getattr(token, "Id", None)
                    try:
                        description = token.GetDescription()
                    except Exception:
                        description = token_id or "未知输出设备"
                    outputs.append((description, token_id))
            self._outputs = outputs
            return outputs
        finally:
            pythoncom.CoUninitialize()

    def list_outputs(self) -> List[Tuple[str, Optional[str]]]:
        with self._lock:
            outputs = self._collect_outputs()
        if not outputs:
            return [("默认输出设备", None)]
        return list(outputs)

    def set_output(self, token_id: Optional[str]) -> None:
        import pythoncom

        pythoncom.CoInitialize()
        try:
            with self._lock:
                if token_id is None:
                    LOGGER.info("[TTS] SAPI 切换到系统默认输出设备")
                    self._voice.AudioOutput = None
                    self._voice.AudioOutputStream = None
                    self._current_output_id = None
                    return
                # 刷新设备列表，避免持有过期的 COM 句柄
                self._collect_outputs()
                target = self._resolve_output_token(token_id)
                if target is None:
                    raise ValueError("未找到对应的扬声器设备。")
                try:
                    setattr(self._voice, "AllowAudioOutputFormatChangesOnNextSet", True)
                except Exception:
                    pass
                try:
                    self._voice.AudioOutput = target
                    self._current_output_id = token_id
                    LOGGER.info("[TTS] SAPI AudioOutput 已切换到: %s", token_id)
                except Exception as exc:
                    LOGGER.error("[TTS] 设置 AudioOutput 失败 (%s): %s", token_id, exc)
                try:
                    self._voice.AudioOutputStream = None
                except Exception:
                    pass
        finally:
            pythoncom.CoUninitialize()

    def _resolve_output_token(self, token_id: str) -> Optional[object]:
        try:
            collection = self._voice.GetAudioOutputs()
        except Exception:
            return None
        if collection:
            for idx in range(collection.Count):
                token = collection.Item(idx)
                if getattr(token, "Id", None) == token_id:
                    return token
        return None

    def list_voices(self) -> List[VoiceOption]:
        options: List[VoiceOption] = []
        for name, token in self._voices:
            token_id = getattr(token, "Id", None)
            language = self._extract_language(token)
            options.append(VoiceOption(name=name, token_id=token_id, language=language))
        return options

    def set_voice(self, token_id: Optional[str]) -> None:
        with self._lock:
            if token_id is None:
                self._voice.Voice = None
                return
            for _, token in self._voices:
                if getattr(token, "Id", None) == token_id:
                    self._voice.Voice = token
                    return
            raise ValueError("未找到对应的音色。")

    def _collect_voices(self) -> List[Tuple[str, object]]:
        voices: List[Tuple[str, object]] = []
        try:
            collection = self._voice.GetVoices()
        except Exception:
            collection = None
        if collection:
            for idx in range(collection.Count):
                token = collection.Item(idx)
                voices.append((token.GetDescription(), token))
        return voices

    def _extract_language(self, token: object) -> str:
        language = ""
        try:
            attr = token.GetAttribute("Language")  # type: ignore[attr-defined]
        except Exception:
            attr = None
        if attr:
            code = attr.split(";")[0]
            try:
                value = int(code, 16)
                import locale

                language = locale.windows_locale.get(value, "")  # type: ignore[attr-defined]
            except Exception:
                language = code
        if not language:
            try:
                language = token.GetAttribute("Name")  # type: ignore[attr-defined]
            except Exception:
                language = ""
        return language.replace("_", "-") if language else "未知语言"

    def set_rate(self, rate: float) -> bool:
        tempo = max(0.5, min(2.0, float(rate)))
        mapped = int(max(-10, min(10, round((tempo - 1.0) * 10))))
        with self._lock:
            try:
                self._voice.Rate = mapped
            except Exception as exc:
                raise RuntimeError(f"无法调整语速: {exc}") from exc
        return True

    def set_volume(self, volume: float) -> bool:
        gain = max(0.0, min(2.0, float(volume)))
        value = int(max(0, min(100, round(gain * 100))))
        with self._lock:
            try:
                self._voice.Volume = value
            except Exception as exc:
                raise RuntimeError(f"无法调整音量: {exc}") from exc
        return True

    def speak(self, text: str) -> None:
        snippet = text.strip()
        if not snippet:
            return

        import pythoncom

        pythoncom.CoInitialize()
        try:
            with self._lock:
                voice = self._voice
                stop_event = self._stop_event
                output_id = self._current_output_id

            def _purge() -> None:
                try:
                    constants = getattr(win32com.client, "constants", None)
                    purge_flag = getattr(constants, "SVSFPurgeBeforeSpeak", 2) if constants is not None else 2
                    voice.Speak("", purge_flag)
                except Exception:
                    pass

            target = None
            stream = None
            if output_id:
                target = self._resolve_output_token(output_id)
                if target is not None:
                    try:
                        voice.AudioOutput = target
                    except Exception:
                        LOGGER.warning("[TTS] 无法设置音频输出令牌: %s", output_id)
                    try:
                        stream = target.CreateInstance()
                    except Exception:
                        stream = None
                    if stream is not None:
                        try:
                            voice.AudioOutputStream = stream
                        except Exception:
                            LOGGER.warning("[TTS] 无法将 AudioOutputStream 绑定到: %s", output_id)

            # SVSFlagsAsync = 1
            flags = 1

            for chunk in self._split_text(snippet, 80):
                if stop_event is not None and stop_event.is_set():
                    _purge()
                    break
                
                try:
                    voice.Speak(chunk, flags)
                except Exception:
                    continue

                # Wait for the chunk to finish, checking stop_event periodically
                while True:
                    if stop_event is not None and stop_event.is_set():
                        _purge()
                        return
                    try:
                        # WaitUntilDone returns True if the speech is done, False if timed out
                        if voice.WaitUntilDone(50):
                            break
                    except Exception:
                        break
        finally:
            pythoncom.CoUninitialize()

    def _split_text(self, text: str, maxlen: int = 80) -> List[str]:
        import re

        tokens = re.split(r"([。！？；;.!?])", text)
        segments: List[str] = []
        buffer = ""
        for token in tokens:
            if not token:
                continue
            buffer += token
            if len(buffer) >= maxlen or token in "。！？；;.!?":
                segments.append(buffer)
                buffer = ""
        if buffer:
            segments.append(buffer)
        return segments

    def stop(self) -> None:
        with self._lock:
            voice = self._voice
        try:
            constants = getattr(win32com.client, "constants", None)
            purge_flag = getattr(constants, "SVSFPurgeBeforeSpeak", 2) if constants is not None else 2
            voice.Speak("", purge_flag)
        except Exception:
            pass

    def attach_stop_event(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event


class PyttsxTTS(BaseTTS):
    def __init__(self) -> None:
        if pyttsx3 is None:
            raise RuntimeError("未安装 pyttsx3，无法使用备用语音合成。")
        self._engine = pyttsx3.init()
        self._voices = self._collect_voices()
        try:
            self._default_voice_id = self._engine.getProperty("voice")
        except Exception:
            self._default_voice_id = None
        try:
            base_rate = self._engine.getProperty("rate")
            self._base_rate = float(base_rate if base_rate else 200.0)
        except Exception:
            self._base_rate = 200.0
        try:
            base_volume = self._engine.getProperty("volume")
            self._base_volume = float(base_volume if base_volume else 1.0)
        except Exception:
            self._base_volume = 1.0
        self._stop_event: Optional[threading.Event] = None

    def set_output(self, token_id: Optional[str]) -> None:
        if token_id:
            raise ValueError("Pyttsx3 后端不支持选择输出设备，请安装 pywin32 使用 Windows SAPI。")

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def shutdown(self) -> None:
        self._engine.stop()

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass

    def attach_stop_event(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event

    def list_voices(self) -> List[VoiceOption]:
        options: List[VoiceOption] = []
        for entry in self._voices:
            try:
                desc = entry.name or entry.id
            except Exception:
                desc = getattr(entry, "id", "未知音色")
            language = self._extract_language(entry)
            options.append(VoiceOption(name=desc, token_id=getattr(entry, "id", None), language=language))
        return options

    def set_voice(self, token_id: Optional[str]) -> None:
        target_id = token_id or self._default_voice_id
        if not target_id:
            return
        try:
            self._engine.setProperty("voice", target_id)
        except Exception as exc:
            raise ValueError(f"无法切换音色: {exc}") from exc

    def set_rate(self, rate: float) -> bool:
        tempo = max(0.5, min(2.0, float(rate)))
        target = int(max(20, min(500, round(self._base_rate * tempo))))
        try:
            self._engine.setProperty("rate", target)
        except Exception as exc:
            raise RuntimeError(f"无法调整语速: {exc}") from exc
        return True

    def set_volume(self, volume: float) -> bool:
        gain = max(0.0, min(2.0, float(volume)))
        # pyttsx3 支持的音量范围为 0.0-1.0，超出部分取最大值
        target = max(0.0, min(1.0, self._base_volume * gain))
        try:
            self._engine.setProperty("volume", target)
        except Exception as exc:
            raise RuntimeError(f"无法调整音量: {exc}") from exc
        return True

    def _collect_voices(self) -> List[object]:
        try:
            return list(self._engine.getProperty("voices"))
        except Exception:
            return []

    def _extract_language(self, voice: object) -> str:
        try:
            languages = getattr(voice, "languages", None)
        except Exception:
            languages = None
        if languages:
            codes: List[str] = []
            for item in languages:
                try:
                    if isinstance(item, bytes):
                        item = item.decode("utf-8", errors="ignore")
                    if isinstance(item, str):
                        item = item.strip()
                        if item.startswith("\x05"):
                            item = item[1:]
                        codes.append(item.replace("_", "-"))
                except Exception:
                    continue
            if codes:
                return ",".join(codes)
        try:
            return getattr(voice, "id", "未知语言")
        except Exception:
            return "未知语言"


class GttsTTS(BaseTTS):
    def __init__(self) -> None:
        if gTTS is None or tts_langs is None:
            raise RuntimeError("需要安装 gTTS 才能使用 Google 语音合成。")
        self._ffmpeg = _find_ffmpeg()
        if not self._ffmpeg:
            raise RuntimeError("gTTS 语音播放需要预先安装 ffmpeg。")
        raw_languages = tts_langs()
        if not raw_languages:
            raise RuntimeError("未能获取 gTTS 支持的语言列表。")
        normalized: Dict[str, str] = {}
        for code, name in raw_languages.items():
            canonical = code.lower()
            if canonical not in normalized:
                normalized[canonical] = name
        self._languages = dict(sorted(normalized.items(), key=lambda item: item[1].lower()))
        self._voice_code: Optional[str] = None
        default = "zh-cn" if "zh-cn" in self._languages else "en"
        self._default_language = default if default in self._languages else next(iter(self._languages))
        self._voice_options: List[VoiceOption] = []
        self._play_speed = 1.0
        self._volume_scale = 1.0
        self._stop_event: Optional[threading.Event] = None

    def list_outputs(self) -> List[Tuple[str, Optional[str]]]:
        return [("默认输出设备", None)]

    def set_output(self, token_id: Optional[str]) -> None:
        if token_id:
            raise ValueError("gTTS 后端不支持选择输出设备。")

    def list_voices(self) -> List[VoiceOption]:
        if not self._voice_options:
            options: List[VoiceOption] = []
            for code, display_name in self._languages.items():
                label = f"{display_name} ({code})"
                options.append(VoiceOption(name=label, token_id=code, language=label))
            self._voice_options = options
        return list(self._voice_options)

    def set_voice(self, token_id: Optional[str]) -> None:
        if token_id is None:
            self._voice_code = None
            return
        canonical = token_id.lower()
        if canonical not in self._languages:
            raise ValueError("未找到对应的语言。")
        self._voice_code = canonical

    def set_rate(self, rate: float) -> bool:
        tempo = max(0.5, min(2.0, float(rate)))
        self._play_speed = tempo
        return True

    def set_volume(self, volume: float) -> bool:
        gain = max(0.0, min(2.5, float(volume)))
        self._volume_scale = gain
        return True

    def speak(self, text: str) -> None:
        snippet = text.strip()
        if not snippet:
            return
        if self._stop_event is not None and self._stop_event.is_set():
            return
        language = (self._voice_code or self._default_language).lower()
        if language not in self._languages:
            language = self._default_language
        buffer = BytesIO()
        try:
            gTTS(text=snippet, lang=language).write_to_fp(buffer)
        except Exception as exc:
            raise RuntimeError(f"gTTS 合成失败: {exc}") from exc
        try:
            # gTTS 输出为 MP3，需要通过 ffmpeg 转换为 WAV 以便后续播放。
            wav_bytes = _convert_mp3_to_wav(
                buffer.getvalue(),
                self._ffmpeg,
                speed=self._play_speed,
                volume=self._volume_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"音频转换失败: {exc}") from exc
        _play_wav_bytes(wav_bytes, stop_event=self._stop_event)

    def shutdown(self) -> None:
        pass

    def stop(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import winsound

            winsound.PlaySound(None, getattr(winsound, "SND_PURGE", 0))
        except Exception:
            pass

    def attach_stop_event(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event


class EdgeTTS(BaseTTS):
    def __init__(self) -> None:
        if edge_tts is None:
            raise RuntimeError("需要安装 edge-tts 才能使用 Microsoft Edge 语音。")
        self._ffmpeg = _find_ffmpeg()
        if not self._ffmpeg:
            raise RuntimeError("Edge TTS 播放需要预先安装 ffmpeg。")
        self._voice_options = self._load_voice_options()
        if not self._voice_options:
            raise RuntimeError("未获取到任何 Edge 语音。")
        self._voice_tokens = {option.token_id for option in self._voice_options if option.token_id}
        self._default_voice = self._select_default_voice()
        self._voice_id: Optional[str] = None
        self._play_speed = 1.0
        self._volume_scale = 1.0
        self._stop_event: Optional[threading.Event] = None

    def list_outputs(self) -> List[Tuple[str, Optional[str]]]:
        return [("默认输出设备", None)]

    def set_output(self, token_id: Optional[str]) -> None:
        if token_id:
            raise ValueError("Edge TTS 后端不支持选择输出设备。")

    def list_voices(self) -> List[VoiceOption]:
        return list(self._voice_options)

    def set_voice(self, token_id: Optional[str]) -> None:
        if token_id is None:
            self._voice_id = None
            return
        if token_id not in self._voice_tokens:
            raise ValueError("未找到对应的音色。")
        self._voice_id = token_id

    def set_rate(self, rate: float) -> bool:
        tempo = max(0.5, min(2.0, float(rate)))
        self._play_speed = tempo
        return True

    def set_volume(self, volume: float) -> bool:
        gain = max(0.0, min(2.5, float(volume)))
        self._volume_scale = gain
        return True

    def speak(self, text: str) -> None:
        snippet = text.strip()
        if not snippet:
            return
        if self._stop_event is not None and self._stop_event.is_set():
            return
        target_voice = self._voice_id or self._default_voice
        if not target_voice:
            raise RuntimeError("当前 Edge TTS 无可用音色。")
        try:
            mp3_bytes = self._run_async(self._synthesize(snippet, target_voice))
        except Exception as exc:
            raise RuntimeError(f"Edge TTS 合成失败: {exc}") from exc
        if not isinstance(mp3_bytes, (bytes, bytearray)):
            raise RuntimeError("Edge TTS 返回了未知的音频格式。")
        try:
            # Edge 默认输出 MP3，同样通过 ffmpeg 转换为 WAV 保持播放方式一致。
            wav_bytes = _convert_mp3_to_wav(
                bytes(mp3_bytes),
                self._ffmpeg,
                speed=self._play_speed,
                volume=self._volume_scale,
            )
        except Exception as exc:
            raise RuntimeError(f"Edge 音频转换失败: {exc}") from exc
        _play_wav_bytes(wav_bytes, stop_event=self._stop_event)

    def shutdown(self) -> None:
        pass

    def stop(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import winsound

            winsound.PlaySound(None, getattr(winsound, "SND_PURGE", 0))
        except Exception:
            pass

    def attach_stop_event(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event

    async def _synthesize(self, text: str, voice: str) -> bytes:
        communicator = edge_tts.Communicate(text=text, voice=voice)
        data = bytearray()
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":
                data.extend(chunk.get("data", b""))
        return bytes(data)

    def _load_voice_options(self) -> List[VoiceOption]:
        try:
            voices = self._run_async(edge_tts.list_voices())
        except Exception as exc:
            raise RuntimeError(f"无法获取 Edge 语音列表: {exc}") from exc
        options: List[VoiceOption] = []
        for entry in voices:
            short_name = entry.get("ShortName")
            if not short_name:
                continue
            local_name = entry.get("LocalName") or entry.get("FriendlyName") or short_name
            locale = entry.get("Locale", "")
            locale_name = entry.get("LocaleName", "")
            language_label = locale_name or locale
            if locale and locale.lower() not in language_label.lower():
                language_label = f"{language_label} ({locale})" if language_label else locale
            display = local_name if local_name else short_name
            if short_name not in display:
                display = f"{display} ({short_name})"
            options.append(VoiceOption(name=display, token_id=short_name, language=language_label))
        options.sort(key=lambda item: item.name.lower())
        return options

    def _select_default_voice(self) -> Optional[str]:
        preferences = ("zh", "en")
        for prefix in preferences:
            for option in self._voice_options:
                lang = option.language.lower()
                if lang.startswith(prefix):
                    return option.token_id
        return self._voice_options[0].token_id if self._voice_options else None

    @staticmethod
    def _run_async(coro: object) -> object:
        if not asyncio.iscoroutine(coro):
            return coro
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)
            loop.close()


@dataclass
class HotkeyAction:
    keys: List[str]
    mouse_buttons: List[str]


@dataclass
class VoiceOption:
    name: str
    token_id: Optional[str]
    language: str


class HotkeyCapture:
    def __init__(
        self,
        on_finish: Callable[[List[str]], None],
        on_cancel: Callable[[], None],
        ignore_duration: float = 0.15,
    ) -> None:
        if pynput_keyboard is None or pynput_mouse is None:
            raise RuntimeError("需要安装 pynput 才能监听热键。")
        self._on_finish = on_finish
        self._on_cancel = on_cancel
        self._ignore_duration = ignore_duration
        self._keyboard_listener: Optional[object] = None
        self._mouse_listener: Optional[object] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._active = False
        self._start_time = 0.0
        self._captured_tokens: List[str] = []
        self._pressed_keys: Set[str] = set()
        self._pressed_buttons: Set[str] = set()
        self._key_token_map: Dict[object, str] = {}
        self._button_token_map: Dict[object, str] = {}

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._keyboard_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
            suppress=False,
        )
        self._mouse_listener = pynput_mouse.Listener(on_click=self._on_mouse_click)
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def stop(self, notify: bool = False) -> None:
        with self._lock:
            already_stopped = self._stopped.is_set()
            self._stopped.set()
        self._stop_listeners()
        if notify and not already_stopped:
            self._on_cancel()

    def is_running(self) -> bool:
        return not self._stopped.is_set()

    def _on_key_press(self, key: object) -> Optional[bool]:
        if self._should_ignore_events():
            return None
        token = self._normalize_key(key)
        if token is None:
            return None
        cancel = False
        with self._lock:
            if self._stopped.is_set():
                return False
            if token == "esc" and not self._active and not self._captured_tokens:
                cancel = True
            else:
                if key in self._key_token_map:
                    return None
                self._active = True
                if token not in self._captured_tokens:
                    self._captured_tokens.append(token)
                self._key_token_map[key] = token
                self._pressed_keys.add(token)
        if cancel:
            self._stopped.set()
            self._stop_listeners()
            self._on_cancel()
            return False
        return None

    def _on_key_release(self, key: object) -> Optional[bool]:
        if self._should_ignore_events():
            return None
        finish_tokens: Optional[List[str]] = None
        with self._lock:
            if self._stopped.is_set():
                return False
            token = self._key_token_map.pop(key, None)
            if token is not None:
                self._pressed_keys.discard(token)
            finish_tokens = self._check_finalize_locked()
        if finish_tokens is not None:
            self._finalize(finish_tokens)
            return False
        return None

    def _on_mouse_click(self, _x: int, _y: int, button: object, pressed: bool) -> Optional[bool]:
        if self._should_ignore_events():
            return None
        token = self._normalize_button(button)
        if token is None:
            return None
        finish_tokens: Optional[List[str]] = None
        with self._lock:
            if self._stopped.is_set():
                return False
            if pressed:
                if button in self._button_token_map:
                    return None
                self._active = True
                if token not in self._captured_tokens:
                    self._captured_tokens.append(token)
                self._button_token_map[button] = token
                self._pressed_buttons.add(token)
            else:
                mapped = self._button_token_map.pop(button, None)
                if mapped is not None:
                    self._pressed_buttons.discard(mapped)
                finish_tokens = self._check_finalize_locked()
        if finish_tokens is not None:
            self._finalize(finish_tokens)
            return False
        return None

    def _check_finalize_locked(self) -> Optional[List[str]]:
        if self._stopped.is_set():
            return None
        if self._active and not self._pressed_keys and not self._pressed_buttons and self._captured_tokens:
            self._stopped.set()
            return list(self._captured_tokens)
        return None

    def _finalize(self, tokens: List[str]) -> None:
        self._stop_listeners()
        self._on_finish(tokens)

    def _stop_listeners(self) -> None:
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass

    def _should_ignore_events(self) -> bool:
        return self._stopped.is_set() or (time.monotonic() - self._start_time) < self._ignore_duration

    def _normalize_key(self, key: object) -> Optional[str]:
        try:
            from pynput.keyboard import Key, KeyCode  # type: ignore
        except ImportError:  # pragma: no cover - guarded earlier
            return None
        normalized: Optional[str] = None
        if isinstance(key, Key):
            mapping = {
                Key.alt: "alt",
                Key.alt_l: "alt",
                Key.alt_r: "alt",
                Key.ctrl: "ctrl",
                Key.ctrl_l: "ctrl",
                Key.ctrl_r: "ctrl",
                Key.shift: "shift",
                Key.shift_l: "shift",
                Key.shift_r: "shift",
                Key.cmd: "win",
                Key.cmd_l: "win",
                Key.cmd_r: "win",
                Key.enter: "enter",
                Key.space: "space",
                Key.tab: "tab",
                Key.backspace: "backspace",
                Key.delete: "delete",
                Key.esc: "esc",
                Key.home: "home",
                Key.end: "end",
                Key.page_up: "page_up",
                Key.page_down: "page_down",
                Key.up: "up",
                Key.down: "down",
                Key.left: "left",
                Key.right: "right",
            }
            normalized = mapping.get(key)
            if normalized is None and key.name:
                normalized = key.name.lower()
        elif isinstance(key, KeyCode):
            if key.char:
                normalized = key.char.lower()
            elif key.vk is not None:
                normalized = f"vk_{key.vk}"
        return normalized

    def _normalize_button(self, button: object) -> Optional[str]:
        try:
            from pynput.mouse import Button  # type: ignore
        except ImportError:  # pragma: no cover - guarded earlier
            return None
        mapping = {
            Button.left: "mouse_left",
            Button.right: "mouse_right",
            Button.middle: "mouse_middle",
        }
        extra_buttons = {
            "mouse_x1": getattr(Button, "x1", None),
            "mouse_x2": getattr(Button, "x2", None),
        }
        for name, attr in extra_buttons.items():
            if attr is not None:
                mapping[attr] = name
        return mapping.get(button)


class PushToTalkListener:
    def __init__(
        self,
        action: HotkeyAction,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        if pynput_keyboard is None:
            raise RuntimeError("需要安装 pynput 才能启用按键录音功能。")
        self._keys = {token.lower() for token in action.keys}
        self._mouse_buttons = {token.lower() for token in action.mouse_buttons}
        self._expected = set()
        self._expected.update(self._keys)
        self._expected.update(self._mouse_buttons)
        if not self._expected:
            raise ValueError("按键录音组合不能为空。")
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._active = False
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._keyboard_listener: Optional[object] = None
        self._mouse_listener: Optional[object] = None

    def start(self) -> None:
        if self._keyboard_listener is not None:
            return
        self._keyboard_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
            suppress=False,
        )
        self._keyboard_listener.start()
        if self._mouse_buttons and pynput_mouse is not None:
            self._mouse_listener = pynput_mouse.Listener(on_click=self._on_mouse_click)
            self._mouse_listener.start()

    def stop(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._active = False
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        if self._mouse_listener is not None:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
            self._mouse_listener = None

    def _normalize_key(self, key: object) -> Optional[str]:
        try:
            from pynput.keyboard import Key, KeyCode  # type: ignore
        except ImportError:
            return None
        if isinstance(key, Key):
            mapping = {
                Key.alt: "alt",
                Key.alt_l: "alt",
                Key.alt_r: "alt",
                Key.ctrl: "ctrl",
                Key.ctrl_l: "ctrl",
                Key.ctrl_r: "ctrl",
                Key.shift: "shift",
                Key.shift_l: "shift",
                Key.shift_r: "shift",
                Key.cmd: "win",
                Key.cmd_l: "win",
                Key.cmd_r: "win",
                Key.enter: "enter",
                Key.space: "space",
                Key.tab: "tab",
                Key.backspace: "backspace",
                Key.delete: "delete",
                Key.esc: "esc",
                Key.home: "home",
                Key.end: "end",
                Key.page_up: "page_up",
                Key.page_down: "page_down",
                Key.up: "up",
                Key.down: "down",
                Key.left: "left",
                Key.right: "right",
            }
            normalized = mapping.get(key)
            if normalized is not None:
                return normalized
            if key.name:
                return key.name.lower()
            return None
        if isinstance(key, KeyCode):
            if key.char:
                return key.char.lower()
            if key.vk is not None:
                return f"vk_{key.vk}"
        return None

    def _normalize_button(self, button: object) -> Optional[str]:
        try:
            from pynput.mouse import Button  # type: ignore
        except ImportError:
            return None
        mapping = {
            Button.left: "mouse_left",
            Button.right: "mouse_right",
            Button.middle: "mouse_middle",
        }
        extra = {
            "mouse_x1": getattr(Button, "x1", None),
            "mouse_x2": getattr(Button, "x2", None),
        }
        for name, attr in extra.items():
            if attr is not None:
                mapping.update({
                    name: attr,
                    attr: name,
                })
        return mapping.get(button)

    def _on_key_press(self, key: object) -> Optional[bool]:
        token = self._normalize_key(key)
        if token is None or token not in self._keys:
            return None
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            if token in self._pressed:
                return None
            self._pressed.add(token)
            callback = self._evaluate_state_locked()
        if callback is not None:
            try:
                callback()
            except Exception:
                LOGGER.exception("按键录音回调执行失败")
        return None

    def _on_key_release(self, key: object) -> Optional[bool]:
        token = self._normalize_key(key)
        if token is None or token not in self._keys:
            return None
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            if token in self._pressed:
                self._pressed.discard(token)
            callback = self._evaluate_state_locked()
        if callback is not None:
            try:
                callback()
            except Exception:
                LOGGER.exception("按键录音回调执行失败")
        return None

    def _on_mouse_click(self, _x: int, _y: int, button: object, pressed: bool) -> Optional[bool]:
        token = self._normalize_button(button)
        if token is None or token not in self._mouse_buttons:
            return None
        callback: Optional[Callable[[], None]] = None
        with self._lock:
            if pressed:
                if token in self._pressed:
                    return None
                self._pressed.add(token)
            else:
                self._pressed.discard(token)
            callback = self._evaluate_state_locked()
        if callback is not None:
            try:
                callback()
            except Exception:
                LOGGER.exception("按键录音回调执行失败")
        return None

    def _evaluate_state_locked(self) -> Optional[Callable[[], None]]:
        should_activate = self._expected.issubset(self._pressed)
        if should_activate and not self._active:
            self._active = True
            return self._on_activate
        if not should_activate and self._active:
            self._active = False
            return self._on_deactivate
        return None
class TTSManager:
    """Maintains a queue of text requiring speech synthesis."""

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
        playback_state_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._status_callback = status_callback
        self._playback_state_callback = playback_state_callback
        self._tts_lock = threading.Lock()
        (
            self._engine_factories,
            self._engine_names,
            self._engine_order,
        ) = self._discover_engine_factories()
        if not self._engine_order:
            raise RuntimeError("未找到可用的语音合成库，请安装 pywin32、edge-tts、pyttsx3 或 gTTS。")
        self._current_engine_id = self._engine_order[0]
        self._tts = self._engine_factories[self._current_engine_id]()
        self._stop_event = threading.Event()
        try:
            self._tts.attach_stop_event(self._stop_event)
        except Exception:
            pass
        self._playback_rate = 1.0
        self._playback_volume = 1.0
        self._rate_warning_sent = False
        self._volume_warning_sent = False
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._hotkey_lock = threading.Lock()
        self._hotkey_action: Optional[HotkeyAction] = None
        self._hotkey_warning_sent = False
        self._voice_token: Optional[str] = None
        self._voice_options: List[VoiceOption] = []
        self._worker = threading.Thread(target=self._loop, name="TTSWorker", daemon=True)
        self._worker.start()

    def _notify_playback_state(self, is_playing: bool) -> None:
        callback = self._playback_state_callback
        if callback is not None:
            try:
                callback(is_playing)
            except Exception:
                LOGGER.exception("播放状态回调执行失败")

    def _discover_engine_factories(
        self,
    ) -> Tuple[Dict[str, Callable[[], BaseTTS]], Dict[str, str], List[str]]:
        factories: Dict[str, Callable[[], BaseTTS]] = {}
        names: Dict[str, str] = {}
        order: List[str] = []

        def register(engine_id: str, display: str, factory: Callable[[], BaseTTS]) -> None:
            # Instantiate once to ensure the engine works before exposing it.
            try:
                backend = factory()
            except Exception:
                return
            try:
                backend.shutdown()
            except Exception:
                pass
            factories[engine_id] = factory
            names[engine_id] = display
            order.append(engine_id)

        if win32com is not None:
            register("sapi", "Windows SAPI", lambda: SapiTTS())
        if edge_tts is not None:
            register("edge", "微软 Edge TTS", lambda: EdgeTTS())
        if pyttsx3 is not None:
            register("pyttsx3", "pyttsx3", lambda: PyttsxTTS())
        if gTTS is not None and tts_langs is not None:
            register("gtts", "Google TTS (gTTS)", lambda: GttsTTS())

        return factories, names, order

    def available_engines(self) -> List[Tuple[str, str]]:
        return [(self._engine_names[engine_id], engine_id) for engine_id in self._engine_order]

    def current_engine(self) -> str:
        return self._current_engine_id

    def set_engine(self, engine_id: str) -> None:
        if engine_id == self._current_engine_id:
            return
        factory = self._engine_factories.get(engine_id)
        if factory is None:
            raise ValueError("不支持的语音引擎。")
        try:
            backend = factory()
        except Exception as exc:
            raise ValueError(f"无法切换到语音引擎: {exc}") from exc
        with self._tts_lock:
            old_backend = self._tts
            self._tts = backend
            self._current_engine_id = engine_id
            self._voice_token = None
            self._rate_warning_sent = False
            self._volume_warning_sent = False
            try:
                self._tts.attach_stop_event(self._stop_event)
            except Exception:
                pass
        self._voice_options = []
        try:
            old_backend.shutdown()
        except Exception:
            pass
        # Reapply playback preferences on the new backend.
        self.set_playback_rate(self._playback_rate)
        self.set_playback_volume(self._playback_volume)

    def available_outputs(self) -> List[Tuple[str, Optional[str]]]:
        with self._tts_lock:
            outputs = self._tts.list_outputs()
        return list(outputs)

    def set_output(self, token_id: Optional[str]) -> None:
        with self._tts_lock:
            self._tts.set_output(token_id)

    def available_voices(self) -> List[VoiceOption]:
        try:
            with self._tts_lock:
                self._voice_options = self._tts.list_voices()
        except NotImplementedError:
            self._voice_options = []
        return self._voice_options

    def set_voice(self, token_id: Optional[str]) -> None:
        try:
            with self._tts_lock:
                self._tts.set_voice(token_id)
        except NotImplementedError:
            if self._status_callback:
                self._status_callback("当前语音库不支持选择音色。")
            return
        except ValueError as exc:
            if self._status_callback:
                self._status_callback(str(exc))
            raise
        self._voice_token = token_id

    def playback_rate(self) -> float:
        with self._tts_lock:
            return self._playback_rate

    def playback_volume(self) -> float:
        with self._tts_lock:
            return self._playback_volume

    def set_playback_rate(self, rate: float) -> None:
        normalized = max(0.5, min(2.0, float(rate)))
        is_default = math.isclose(normalized, 1.0, rel_tol=1e-3)
        warning: Optional[str] = None
        error: Optional[str] = None
        with self._tts_lock:
            self._playback_rate = normalized
            if is_default:
                self._rate_warning_sent = False
            try:
                supported = self._tts.set_rate(normalized)
            except NotImplementedError:
                supported = False
            except Exception as exc:
                error = f"语速设置失败: {exc}"
                supported = True
            supported = bool(supported)
            if not supported and not is_default and not self._rate_warning_sent:
                warning = "当前语音库不支持调整语速。"
                self._rate_warning_sent = True
        callback = self._status_callback
        if callback:
            if error:
                callback(error)
            elif warning:
                callback(warning)

    def set_playback_volume(self, volume: float) -> None:
        normalized = max(0.0, min(2.5, float(volume)))
        is_default = math.isclose(normalized, 1.0, rel_tol=1e-3)
        warning: Optional[str] = None
        error: Optional[str] = None
        with self._tts_lock:
            self._playback_volume = normalized
            if is_default:
                self._volume_warning_sent = False
            try:
                supported = self._tts.set_volume(normalized)
            except NotImplementedError:
                supported = False
            except Exception as exc:
                error = f"音量设置失败: {exc}"
                supported = True
            supported = bool(supported)
            if not supported and not is_default and not self._volume_warning_sent:
                warning = "当前语音库不支持调整音量。"
                self._volume_warning_sent = True
        callback = self._status_callback
        if callback:
            if error:
                callback(error)
            elif warning:
                callback(warning)

    def set_hotkey(self, action: Optional[HotkeyAction]) -> None:
        with self._hotkey_lock:
            self._hotkey_action = action
            self._hotkey_warning_sent = False

    def enqueue(self, text: str) -> None:
        trimmed = text.strip()
        if trimmed:
            self._queue.put(trimmed)

    def test(self) -> None:
        self.enqueue("这是语音合成测试。")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=1)
        with self._tts_lock:
            backend = self._tts
        try:
            backend.shutdown()
        except Exception:
            pass

    def emergency_stop(self) -> None:
        self._stop_event.set()
        sentinel_found = self._drain_queue()
        # Wake worker thread so it can observe the stop event.
        try:
            self._queue.put_nowait("")
        except queue.Full:  # pragma: no cover - unbounded queue
            pass
        if sentinel_found:
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover - unbounded queue
                pass
        with self._tts_lock:
            backend = self._tts
        stopper = getattr(backend, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except Exception as exc:  # pragma: no cover - backend specific
                if self._status_callback is not None:
                    self._status_callback(f"急停停止朗读失败: {exc}")
        self._notify_playback_state(False)

    def _drain_queue(self) -> bool:
        sentinel_found = False
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    sentinel_found = True
        except queue.Empty:
            pass
        return sentinel_found

    def _loop(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                self._notify_playback_state(False)
                break
            if self._stop_event.is_set():
                self._stop_event.clear()
                continue
            if not text:
                continue
            release_hotkey: Optional[Callable[[], None]] = None
            started = False
            backend: Optional[BaseTTS]
            backend = None
            try:
                release_hotkey = self._press_hotkey()
                with self._tts_lock:
                    if self._stop_event.is_set():
                        self._stop_event.clear()
                    else:
                        backend = self._tts
                if backend is None:
                    continue
                if self._stop_event.is_set():
                    self._stop_event.clear()
                    continue
                started = True
                self._notify_playback_state(True)
                backend.speak(text)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"TTS 出错: {exc}")
            finally:
                if started:
                    self._notify_playback_state(False)
                if release_hotkey is not None:
                    try:
                        release_hotkey()
                    except Exception as exc:
                        if self._status_callback:
                            self._status_callback(f"热键释放失败: {exc}")
            if self._stop_event.is_set():
                self._stop_event.clear()

    def _press_hotkey(self) -> Optional[Callable[[], None]]:
        with self._hotkey_lock:
            action = self._hotkey_action
        if action is None:
            return None
        description = self._describe_hotkey(action)
        errors: List[str] = []
        if pynput_keyboard is not None and pynput_mouse is not None:
            try:
                LOGGER.info("尝试通过 pynput 触发热键: %s", description)
                releaser = self._press_hotkey_pynput(action)
                LOGGER.info("pynput 已按下热键: %s", description)
                return releaser
            except Exception as exc:
                LOGGER.warning("pynput 热键按下失败: %s (%s)", description, exc)
                errors.append(f"pynput: {exc}")
        release_pyautogui, pyautogui_error = self._press_hotkey_pyautogui(action)
        if release_pyautogui is not None:
            LOGGER.info("通过 pyautogui 已按下热键: %s", description)
            return release_pyautogui
        if pyautogui_error:
            LOGGER.warning("pyautogui 热键按下失败: %s (%s)", description, pyautogui_error)
            errors.append(f"pyautogui: {pyautogui_error}")
        release_native, native_error = self._press_hotkey_native(action)
        if release_native is not None:
            LOGGER.info("通过原生 SendInput 已按下热键: %s", description)
            return release_native
        if native_error:
            LOGGER.warning("原生热键按下失败: %s (%s)", description, native_error)
            errors.append(f"native: {native_error}")
        if errors and self._status_callback and not self._hotkey_warning_sent:
            detail = "; ".join(errors)
            self._status_callback(
                f"热键触发失败 ({detail})，可尝试安装 pynput/pyautogui 或以管理员身份运行程序。"
            )
            self._hotkey_warning_sent = True
        LOGGER.error("所有方式均未能按下热键: %s", description)
        return None

    def _press_hotkey_pyautogui(self, action: HotkeyAction) -> Tuple[Optional[Callable[[], None]], Optional[str]]:
        if pyautogui is None:
            return None, "未安装 pyautogui"
        unsupported = [btn for btn in action.mouse_buttons if btn not in {"left", "right", "middle"}]
        if unsupported:
            detail = ", ".join(unsupported)
            return None, f"鼠标侧键不受支持: {detail}"
        if not action.keys and not action.mouse_buttons:
            return None, "未配置任何按键"
        pressed_keys: List[str] = []
        pressed_buttons: List[str] = []
        try:
            if action.mouse_buttons:
                for key in action.keys:
                    pyautogui.keyDown(key)
                    pressed_keys.append(key)
                for button in action.mouse_buttons:
                    pyautogui.mouseDown(button=button)
                    pressed_buttons.append(button)
            else:
                for key in action.keys:
                    pyautogui.keyDown(key)
                    pressed_keys.append(key)
        except Exception as exc:
            for button in reversed(pressed_buttons):
                try:
                    pyautogui.mouseUp(button=button)
                except Exception:
                    pass
            for key in reversed(pressed_keys):
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass
            return None, str(exc)

        description = self._describe_hotkey(action)

        def _release() -> None:
            LOGGER.info("释放 pyautogui 热键: %s", description)
            for button in reversed(pressed_buttons):
                try:
                    pyautogui.mouseUp(button=button)
                except Exception:
                    pass
            for key in reversed(pressed_keys):
                try:
                    pyautogui.keyUp(key)
                except Exception:
                    pass

        return _release, None

    def _press_hotkey_native(self, action: HotkeyAction) -> Tuple[Optional[Callable[[], None]], Optional[str]]:
        if _USER32 is None:
            return None, "原生输入仅在 Windows 可用"
        if action.mouse_buttons:
            return None, "原生输入暂不支持鼠标按键"
        if not action.keys:
            return None, "未配置任何键盘按键"
        vk_codes: List[int] = []
        inputs_down: List[object] = []
        try:
            for token in action.keys:
                code = self._resolve_vk_code(token)
                if code is None:
                    raise ValueError(f"无法识别的键: {token}")
                vk_codes.append(code)
                inputs_down.append(_build_keyboard_input(code, True))
            _send_keyboard_inputs(inputs_down)
        except Exception as exc:
            return None, str(exc)

        description = self._describe_hotkey(action)

        def _release() -> None:
            inputs_up = [_build_keyboard_input(code, False) for code in reversed(vk_codes)]
            try:
                _send_keyboard_inputs(inputs_up)
                LOGGER.info("释放原生热键: %s", description)
            except Exception:
                # Propagate exception so caller can surface warning
                raise

        return _release, None

    def _resolve_vk_code(self, token: str) -> Optional[int]:
        canonical = token.strip().lower().replace(" ", "_")
        mapping = {
            "ctrl": 0x11,
            "control": 0x11,
            "alt": 0x12,
            "shift": 0x10,
            "win": 0x5B,
            "cmd": 0x5B,
            "meta": 0x5B,
            "super": 0x5B,
            "enter": 0x0D,
            "return": 0x0D,
            "space": 0x20,
            "tab": 0x09,
            "backspace": 0x08,
            "delete": 0x2E,
            "del": 0x2E,
            "insert": 0x2D,
            "esc": 0x1B,
            "escape": 0x1B,
            "home": 0x24,
            "end": 0x23,
            "page_up": 0x21,
            "page-down": 0x22,
            "page_down": 0x22,
            "up": 0x26,
            "down": 0x28,
            "left": 0x25,
            "right": 0x27,
            "capslock": 0x14,
            "caps_lock": 0x14,
            "menu": 0x5D,
            "printscreen": 0x2C,
            "scroll_lock": 0x91,
            "pause": 0x13,
        }
        resolved = mapping.get(canonical)
        if resolved is not None:
            return resolved
        if canonical.startswith("vk_"):
            try:
                return int(canonical[3:])
            except ValueError:
                return None
        normalized = canonical.replace("_", "")
        if normalized.startswith("f") and normalized[1:].isdigit():
            index = int(normalized[1:])
            if 1 <= index <= 24:
                return 0x70 + index - 1
        if len(normalized) == 1:
            return ord(normalized.upper())
        return None

    def _press_hotkey_pynput(self, action: HotkeyAction) -> Callable[[], None]:
        try:
            from pynput.keyboard import Controller as KeyboardController, Key, KeyCode  # type: ignore
            from pynput.mouse import Controller as MouseController, Button  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded earlier
            raise RuntimeError("未安装 pynput") from exc

        keyboard = KeyboardController()
        mouse = MouseController()
        pressed_keys: List[object] = []
        pressed_buttons: List[object] = []
        description = self._describe_hotkey(action)
        try:
            for token in action.keys:
                key_obj = self._resolve_key_token(token)
                if key_obj is None:
                    raise ValueError(f"无法识别的键: {token}")
                keyboard.press(key_obj)
                pressed_keys.append(key_obj)
            for token in action.mouse_buttons:
                button_obj = self._resolve_mouse_button(token)
                if button_obj is None:
                    raise ValueError(f"无法识别的鼠标按键: {token}")
                mouse.press(button_obj)
                pressed_buttons.append(button_obj)
        except Exception as exc:
            for button in reversed(pressed_buttons):
                try:
                    mouse.release(button)
                except Exception:
                    pass
            for key_obj in reversed(pressed_keys):
                try:
                    keyboard.release(key_obj)
                except Exception:
                    pass
            raise
        if not pressed_keys and not pressed_buttons:
            raise ValueError("未配置任何按键")

        def _release() -> None:
            LOGGER.info("释放 pynput 热键: %s", description)
            for button in reversed(pressed_buttons):
                try:
                    mouse.release(button)
                except Exception:
                    pass
            for key_obj in reversed(pressed_keys):
                try:
                    keyboard.release(key_obj)
                except Exception:
                    pass

        return _release

    def _describe_hotkey(self, action: HotkeyAction) -> str:
        parts: List[str] = []
        if action.keys:
            parts.append("+".join(action.keys))
        if action.mouse_buttons:
            parts.append("+".join(action.mouse_buttons))
        return " + ".join(parts) if parts else "<未配置>"

    def _resolve_key_token(self, token: str) -> Optional[object]:
        try:
            from pynput.keyboard import Key, KeyCode  # type: ignore
        except ImportError:  # pragma: no cover - guarded earlier
            return None
        canonical = token.strip().lower().replace(" ", "_")
        mapping = {
            "ctrl": Key.ctrl,
            "control": Key.ctrl,
            "alt": Key.alt,
            "shift": Key.shift,
            "win": getattr(Key, "cmd", None),
            "cmd": getattr(Key, "cmd", None),
            "meta": getattr(Key, "cmd", None),
            "super": getattr(Key, "cmd", None),
            "enter": Key.enter,
            "return": Key.enter,
            "space": Key.space,
            "tab": Key.tab,
            "backspace": Key.backspace,
            "delete": Key.delete,
            "del": Key.delete,
            "insert": Key.insert,
            "esc": Key.esc,
            "escape": Key.esc,
            "home": Key.home,
            "end": Key.end,
            "page_up": Key.page_up,
            "page-down": Key.page_down,
            "page_down": Key.page_down,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "capslock": getattr(Key, "caps_lock", None),
            "caps_lock": getattr(Key, "caps_lock", None),
            "menu": getattr(Key, "menu", None),
            "printscreen": getattr(Key, "print_screen", None),
            "scroll_lock": getattr(Key, "scroll_lock", None),
            "pause": getattr(Key, "pause", None),
        }
        resolved = mapping.get(canonical)
        if resolved is not None:
            return resolved
        if canonical.startswith("vk_"):
            try:
                vk_value = int(canonical[3:])
            except ValueError:
                return None
            return KeyCode.from_vk(vk_value)
        func_key = canonical.replace("_", "")
        if func_key.startswith("f") and func_key[1:].isdigit():
            attr_name = func_key
            resolved = getattr(Key, attr_name, None)
            if resolved is not None:
                return resolved
        if len(canonical) == 1:
            return KeyCode.from_char(canonical)
        sanitized = canonical.replace("_", "")
        if len(sanitized) == 1:
            return KeyCode.from_char(sanitized)
        return None

    def _resolve_mouse_button(self, token: str) -> Optional[object]:
        try:
            from pynput.mouse import Button  # type: ignore
        except ImportError:  # pragma: no cover - guarded earlier
            return None
        canonical = token.strip().lower().replace("-", "_").replace(" ", "")
        mapping = {
            "left": Button.left,
            "mouse_left": Button.left,
            "mouseleft": Button.left,
            "right": Button.right,
            "mouse_right": Button.right,
            "mouseright": Button.right,
            "middle": Button.middle,
            "mouse_middle": Button.middle,
            "mousemiddle": Button.middle,
        }
        x1 = getattr(Button, "x1", None)
        if x1 is not None:
            mapping.update({
                "x1": x1,
                "mouse_x1": x1,
                "mousex1": x1,
            })
        x2 = getattr(Button, "x2", None)
        if x2 is not None:
            mapping.update({
                "x2": x2,
                "mouse_x2": x2,
                "mousex2": x2,
            })
        return mapping.get(canonical)


class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("实时语音转写与朗读")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.local_recognition = RecognitionController(self)
        self.remote_recognition: Optional[RemoteRecognitionController] = None
        self.active_controller: Optional[object] = self.local_recognition
        self.tts_manager: Optional[TTSManager] = None
        self.microphone_map: Dict[str, Optional[int]] = {}
        self.engine_map: Dict[str, str] = {}
        self.speaker_map: Dict[str, Optional[str]] = {}
        self.voice_map: Dict[str, Optional[str]] = {}
        self.language_map: Dict[str, Optional[str]] = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self._settings_path = os.path.join(base_dir, "vtt-settings.json")
        self._remote_test_wave_lock = threading.Lock()
        self._settings_save_job: Optional[str] = None
        self._applying_preferences = False
        self._setting_traces_installed = False
        self._setting_trace_callbacks: List[Callable[..., None]] = []
        self._setting_trace_ids: List[Tuple[tk.Variable, str]] = []
        self._pending_microphone_name: Optional[str] = None
        self._pending_engine_id: Optional[str] = None
        self._pending_speaker_token: Optional[str] = None
        self._pending_speaker_token_set = False
        self._pending_voice_token: Optional[str] = None
        self._pending_voice_token_set = False
        self._pending_language_label: Optional[str] = None
        self._pending_language_value: Optional[str] = None
        self._pending_language_set = False
        self._pending_ptt_enabled = False
        self._pending_show_overlay = False
        self._stopping_recognition = False

        self.model_path_var = tk.StringVar(value="vosk-model")
        self.microphone_var = tk.StringVar()
        self.engine_var = tk.StringVar()
        self.speaker_var = tk.StringVar()
        self.voice_var = tk.StringVar()
        self.language_var = tk.StringVar(value="全部语言")
        self.partial_var = tk.StringVar()
        self.status_var = tk.StringVar(value="准备就绪")
        self.hotkey_var = tk.StringVar()
        self.inference_mode_var = tk.StringVar(value="本地推理")
        self.server_host_var = tk.StringVar(value="127.0.0.1")
        self.server_port_var = tk.StringVar(value="8765")
        self.tts_rate_var = tk.DoubleVar(value=1.0)
        self.tts_volume_var = tk.DoubleVar(value=1.0)
        self._last_inference_mode = self.inference_mode_var.get()
        self._hotkey_capture: Optional[HotkeyCapture] = None
        self._ptt_hotkey_capture: Optional[HotkeyCapture] = None
        self.show_overlay_var = tk.BooleanVar(value=False)
        self.overlay_window: Optional[tk.Toplevel] = None
        self.overlay_text_var = tk.StringVar(value="等待识别...")
        self.overlay_status_var = tk.StringVar(value="○ 空闲")
        self.overlay_recording_var = tk.StringVar(value="○ 未录音")
        self.overlay_recording_label: Optional[tk.Label] = None
        self._overlay_drag_offset = (0, 0)
        self._recording_indicator_color = "#8ec07c"
        self._remote_test_thread: Optional[threading.Thread] = None
        self._overlay_last_text = ""
        self.ptt_enabled_var = tk.BooleanVar(value=False)
        self.ptt_hotkey_var = tk.StringVar()
        self._ptt_listener: Optional[PushToTalkListener] = None
        self._ptt_active = False
        self._ptt_starting = False
        self._ptt_stop_pending = False
        self._ptt_action: Optional[HotkeyAction] = None
        self.emergency_hotkey_var = tk.StringVar()
        self._emergency_hotkey_capture: Optional[HotkeyCapture] = None
        self._emergency_listener: Optional[PushToTalkListener] = None
        self._emergency_action: Optional[HotkeyAction] = None
        self._rate_scale: Optional[ttk.Scale] = None
        self._volume_scale: Optional[ttk.Scale] = None
        self._rate_value_label: Optional[ttk.Label] = None
        self._volume_value_label: Optional[ttk.Label] = None
        self._model_preload_job: Optional[str] = None
        self._is_tts_playing = False
        self._is_recording = False

        self._final_text = ""
        self._spoken_offset = 0
        self._last_final_segment = ""
        self._last_partial_raw = ""
        self._truncation_notified = False
        self._final_history_limit = 8000

        self._load_settings()
        self._init_tts()
        self._build_ui()
        self.refresh_devices()
        self._apply_post_load_preferences()
        self._install_setting_traces()
        self._maybe_preload_local_model()
        self._apply_emergency_hotkey(show_errors=False)
        self._set_overlay_recording_state("idle")

    def _init_tts(self) -> None:
        try:
            self.tts_manager = TTSManager(
                status_callback=self.post_status,
                playback_state_callback=self._on_tts_playback_state,
            )
            try:
                hotkey_action = self._parse_hotkey(self.hotkey_var.get())
            except ValueError as exc:
                self.post_status(f"热键未应用: {exc}")
                hotkey_action = None
            self.tts_manager.set_hotkey(hotkey_action)
            self.tts_manager.set_playback_rate(self.tts_rate_var.get())
            self.tts_manager.set_playback_volume(self.tts_volume_var.get())
            self.overlay_status_var.set("○ 空闲")
        except RuntimeError as exc:
            messagebox.showerror("语音合成不可用", str(exc))
            self.tts_manager = None
            self.overlay_status_var.set("× 未初始化")

    def _load_settings(self) -> None:
        try:
            with open(self._settings_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("设置加载失败: %s", exc)
            return

        self.model_path_var.set(data.get("model_path", self.model_path_var.get()))
        inference_mode = data.get("inference_mode")
        if inference_mode in {"本地推理", "服务器推理"}:
            self.inference_mode_var.set(inference_mode)
            self._last_inference_mode = inference_mode
        self.server_host_var.set(data.get("server_host", self.server_host_var.get()))
        self.server_port_var.set(data.get("server_port", self.server_port_var.get()))
        self.hotkey_var.set(data.get("hotkey", self.hotkey_var.get()))
        self.ptt_hotkey_var.set(data.get("ptt_hotkey", self.ptt_hotkey_var.get()))
        self.emergency_hotkey_var.set(data.get("emergency_hotkey", self.emergency_hotkey_var.get()))
        rate_value = data.get("tts_rate")
        try:
            if rate_value is not None:
                rate_numeric = float(rate_value)
                self.tts_rate_var.set(max(0.5, min(2.0, rate_numeric)))
        except (TypeError, ValueError):
            pass
        volume_value = data.get("tts_volume")
        try:
            if volume_value is not None:
                volume_numeric = float(volume_value)
                self.tts_volume_var.set(max(0.0, min(2.5, volume_numeric)))
        except (TypeError, ValueError):
            pass
        self._pending_microphone_name = data.get("microphone_name")
        self._pending_engine_id = data.get("tts_engine_id")
        if "speaker_token" in data:
            self._pending_speaker_token = data.get("speaker_token")
            self._pending_speaker_token_set = True
        if "voice_token" in data:
            self._pending_voice_token = data.get("voice_token")
            self._pending_voice_token_set = True
        if "language_value" in data:
            self._pending_language_value = data.get("language_value")
            self._pending_language_label = data.get("language_label")
            self._pending_language_set = True
        self.language_var.set(data.get("language_label", self.language_var.get()))
        self.microphone_var.set(self._pending_microphone_name or self.microphone_var.get())
        self.engine_var.set(data.get("tts_engine_label", self.engine_var.get()))
        self.speaker_var.set(data.get("speaker_label", self.speaker_var.get()))
        self.voice_var.set(data.get("voice_label", self.voice_var.get()))
        show_overlay = bool(data.get("show_overlay", False))
        self.show_overlay_var.set(show_overlay)
        self._pending_show_overlay = show_overlay
        ptt_enabled = bool(data.get("ptt_enabled", False))
        self.ptt_enabled_var.set(ptt_enabled)
        self._pending_ptt_enabled = ptt_enabled

    def _apply_post_load_preferences(self) -> None:
        if self.show_overlay_var.get():
            self._ensure_overlay()
        if self._pending_ptt_enabled and not self._is_recognition_running():
            self._applying_preferences = True
            try:
                self.ptt_enabled_var.set(True)
                self._toggle_ptt()
            finally:
                self._applying_preferences = False
            self._pending_ptt_enabled = False
        if self._pending_show_overlay:
            self._pending_show_overlay = False

    def _install_setting_traces(self) -> None:
        if self._setting_traces_installed:
            return

        def register(var: tk.Variable) -> None:
            callback = lambda *_args: self._on_setting_var_changed()
            self._setting_trace_callbacks.append(callback)
            trace_id = var.trace_add("write", callback)
            self._setting_trace_ids.append((var, trace_id))

        for variable in (self.model_path_var, self.server_host_var, self.server_port_var):
            register(variable)
        # 额外监听模型路径变化以便后台预加载本地模型。
        def _model_trace_callback(*_args: object) -> None:
            self._on_model_path_changed()

        trace_id = self.model_path_var.trace_add("write", _model_trace_callback)
        self._setting_trace_callbacks.append(_model_trace_callback)
        self._setting_trace_ids.append((self.model_path_var, trace_id))
        self._setting_traces_installed = True

    def _on_setting_var_changed(self) -> None:
        self._schedule_settings_save()

    def _on_model_path_changed(self) -> None:
        self._schedule_model_preload()

    def _schedule_settings_save(self, force: bool = False) -> None:
        if getattr(self, "root", None) is None:
            return
        if self._applying_preferences and not force:
            return
        if self._settings_save_job is not None:
            try:
                self.root.after_cancel(self._settings_save_job)
            except Exception:  # pragma: no cover - defensive
                pass
        self._settings_save_job = self.root.after(800, self._save_settings)

    def _schedule_model_preload(self) -> None:
        if getattr(self, "root", None) is None:
            return
        if self._model_preload_job is not None:
            try:
                self.root.after_cancel(self._model_preload_job)
            except Exception:
                pass
            self._model_preload_job = None
        self._model_preload_job = self.root.after(600, self._maybe_preload_local_model)

    def _maybe_preload_local_model(self) -> None:
        self._model_preload_job = None
        if self.inference_mode_var.get() != "本地推理":
            return
        model_path = self.model_path_var.get().strip()
        if not model_path:
            return
        try:
            self.local_recognition.preload_model(model_path)
        except Exception as exc:  # pragma: no cover - defensive
            self.post_status(f"模型预加载失败: {exc}")

    def _save_settings(self) -> None:
        self._settings_save_job = None
        data: Dict[str, object] = {
            "model_path": self.model_path_var.get().strip(),
            "inference_mode": self.inference_mode_var.get(),
            "server_host": self.server_host_var.get().strip(),
            "server_port": self.server_port_var.get().strip(),
            "hotkey": self.hotkey_var.get().strip(),
            "ptt_hotkey": self.ptt_hotkey_var.get().strip(),
            "emergency_hotkey": self.emergency_hotkey_var.get().strip(),
            "microphone_name": self.microphone_var.get().strip(),
            "show_overlay": bool(self.show_overlay_var.get()),
            "ptt_enabled": bool(self.ptt_enabled_var.get()),
            "tts_rate": float(max(0.5, min(2.0, self.tts_rate_var.get()))),
            "tts_volume": float(max(0.0, min(2.5, self.tts_volume_var.get()))),
        }
        engine_label = self.engine_var.get().strip()
        data["tts_engine_label"] = engine_label
        speaker_label = self.speaker_var.get().strip()
        data["speaker_label"] = speaker_label
        voice_label = self.voice_var.get().strip()
        data["voice_label"] = voice_label
        language_label = self.language_var.get().strip()
        data["language_label"] = language_label
        if self.tts_manager is not None:
            try:
                data["tts_engine_id"] = self.tts_manager.current_engine()
            except Exception:
                data["tts_engine_id"] = None
        else:
            data["tts_engine_id"] = None
        if speaker_label in self.speaker_map:
            data["speaker_token"] = self.speaker_map.get(speaker_label)
        if voice_label in self.voice_map:
            data["voice_token"] = self.voice_map.get(voice_label)
        if language_label in self.language_map:
            data["language_value"] = self.language_map.get(language_label)
        try:
            with open(self._settings_path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("设置保存失败: %s", exc)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = ttk.Frame(main)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Vosk 模型目录:").pack(side=tk.LEFT)
        self.model_path_entry = ttk.Entry(row, textvariable=self.model_path_var, width=40)
        self.model_path_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.model_browse_button = ttk.Button(row, text="浏览", command=self._select_model_path)
        self.model_browse_button.pack(side=tk.LEFT)

        mode_row = ttk.Frame(main)
        mode_row.pack(fill=tk.X, pady=4)
        ttk.Label(mode_row, text="推理模式:").pack(side=tk.LEFT)
        self.mode_combo = ttk.Combobox(
            mode_row,
            textvariable=self.inference_mode_var,
            state="readonly",
            values=("本地推理", "服务器推理"),
            width=12,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=4)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_inference_mode_changed)

        server_row = ttk.Frame(main)
        server_row.pack(fill=tk.X, pady=4)
        ttk.Label(server_row, text="服务器地址:").pack(side=tk.LEFT)
        self.server_host_entry = ttk.Entry(server_row, textvariable=self.server_host_var, width=18)
        self.server_host_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(server_row, text="端口:").pack(side=tk.LEFT)
        self.server_port_entry = ttk.Entry(server_row, textvariable=self.server_port_var, width=8)
        self.server_port_entry.pack(side=tk.LEFT, padx=4)
        self.remote_test_button = ttk.Button(server_row, text="测试服务器", command=self._test_remote_server)
        self.remote_test_button.pack(side=tk.LEFT, padx=4)

        mic_row = ttk.Frame(main)
        mic_row.pack(fill=tk.X, pady=4)
        ttk.Label(mic_row, text="麦克风:").pack(side=tk.LEFT)
        self.mic_combo = ttk.Combobox(mic_row, textvariable=self.microphone_var, state="readonly")
        self.mic_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_microphone_selected)

        engine_row = ttk.Frame(main)
        engine_row.pack(fill=tk.X, pady=4)
        ttk.Label(engine_row, text="引擎:").pack(side=tk.LEFT)
        self.engine_combo = ttk.Combobox(engine_row, textvariable=self.engine_var, state="readonly")
        self.engine_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)

        spk_row = ttk.Frame(main)
        spk_row.pack(fill=tk.X, pady=4)
        ttk.Label(spk_row, text="扬声器:").pack(side=tk.LEFT)
        self.speaker_combo = ttk.Combobox(spk_row, textvariable=self.speaker_var, state="readonly")
        self.speaker_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.speaker_combo.bind("<<ComboboxSelected>>", self._on_speaker_selected)

        voice_row = ttk.Frame(main)
        voice_row.pack(fill=tk.X, pady=4)
        ttk.Label(voice_row, text="语言:").pack(side=tk.LEFT)
        self.language_combo = ttk.Combobox(voice_row, textvariable=self.language_var, state="readonly", width=12)
        self.language_combo.pack(side=tk.LEFT, padx=4)
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)
        ttk.Label(voice_row, text="音色:").pack(side=tk.LEFT)
        self.voice_combo = ttk.Combobox(voice_row, textvariable=self.voice_var, state="readonly")
        self.voice_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.voice_combo.bind("<<ComboboxSelected>>", self._on_voice_selected)

        rate_row = ttk.Frame(main)
        rate_row.pack(fill=tk.X, pady=4)
        ttk.Label(rate_row, text="朗读速度:").pack(side=tk.LEFT)
        self._rate_scale = ttk.Scale(
            rate_row,
            variable=self.tts_rate_var,
            from_=0.5,
            to=2.0,
            orient=tk.HORIZONTAL,
            command=self._on_rate_slider_changed,
        )
        self._rate_scale.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._rate_value_label = ttk.Label(rate_row, width=7)
        self._rate_value_label.pack(side=tk.LEFT, padx=(4, 0))

        volume_row = ttk.Frame(main)
        volume_row.pack(fill=tk.X, pady=4)
        ttk.Label(volume_row, text="朗读音量:").pack(side=tk.LEFT)
        self._volume_scale = ttk.Scale(
            volume_row,
            variable=self.tts_volume_var,
            from_=0.0,
            to=2.5,
            orient=tk.HORIZONTAL,
            command=self._on_volume_slider_changed,
        )
        self._volume_scale.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._volume_value_label = ttk.Label(volume_row, width=7)
        self._volume_value_label.pack(side=tk.LEFT, padx=(4, 0))

        self._update_rate_label()
        self._update_volume_label()

        hotkey_row = ttk.Frame(main)
        hotkey_row.pack(fill=tk.X, pady=4)
        ttk.Label(hotkey_row, text="朗读热键:").pack(side=tk.LEFT)
        self.hotkey_entry = ttk.Entry(hotkey_row, textvariable=self.hotkey_var, width=30)
        self.hotkey_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.hotkey_entry.bind("<Return>", lambda event: self._apply_hotkey())
        self.hotkey_capture_button = ttk.Button(hotkey_row, text="录制", command=self._capture_hotkey)
        self.hotkey_capture_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(hotkey_row, text="应用", command=self._apply_hotkey).pack(side=tk.LEFT, padx=4)
        ttk.Button(hotkey_row, text="清除", command=self._clear_hotkey).pack(side=tk.LEFT, padx=4)

        ptt_row = ttk.Frame(main)
        ptt_row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(
            ptt_row,
            text="按键录音",
            variable=self.ptt_enabled_var,
            command=self._toggle_ptt,
        ).pack(side=tk.LEFT)
        self.ptt_entry = ttk.Entry(ptt_row, textvariable=self.ptt_hotkey_var, width=30, state="readonly")
        self.ptt_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.ptt_capture_button = ttk.Button(ptt_row, text="设置按键", command=self._capture_ptt_hotkey)
        self.ptt_capture_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(ptt_row, text="清除", command=self._clear_ptt_hotkey).pack(side=tk.LEFT, padx=4)

        emergency_row = ttk.Frame(main)
        emergency_row.pack(fill=tk.X, pady=4)
        ttk.Label(emergency_row, text="急停按钮:").pack(side=tk.LEFT)
        self.emergency_entry = ttk.Entry(
            emergency_row,
            textvariable=self.emergency_hotkey_var,
            state="readonly",
            width=30,
        )
        self.emergency_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self.emergency_capture_button = ttk.Button(
            emergency_row,
            text="设置按键",
            command=self._capture_emergency_hotkey,
        )
        self.emergency_capture_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(emergency_row, text="清除", command=self._clear_emergency_hotkey).pack(side=tk.LEFT, padx=4)

        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X, pady=8)
        ttk.Button(btn_row, text="刷新设备", command=self.refresh_devices).pack(side=tk.LEFT)
        self.start_button = ttk.Button(btn_row, text="开始识别", command=self.start_recognition)
        self.start_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(btn_row, text="停止", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="急停", command=self._on_emergency_button).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="测试朗读", command=self._test_tts).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            btn_row,
            text="悬浮窗",
            variable=self.show_overlay_var,
            command=self._toggle_overlay,
        ).pack(side=tk.LEFT, padx=4)

        text_frame = ttk.Frame(main)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.text_widget = tk.Text(text_frame, height=12, wrap=tk.WORD)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_widget.configure(state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(text_frame, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        partial_row = ttk.Frame(main)
        partial_row.pack(fill=tk.X, pady=4)
        ttk.Label(partial_row, text="实时识别:").pack(side=tk.LEFT)
        ttk.Label(partial_row, textvariable=self.partial_var, foreground="#007acc").pack(side=tk.LEFT, padx=4)

        status_bar = ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(8, 0))
        self._update_server_controls()

    def _update_rate_label(self, value: Optional[float] = None) -> None:
        try:
            current = float(self.tts_rate_var.get()) if value is None else float(value)
        except (tk.TclError, TypeError, ValueError):
            current = 1.0
        display = f"{max(0.5, min(2.0, current)):.2f}x"
        if self._rate_value_label is not None:
            self._rate_value_label.configure(text=display)

    def _update_volume_label(self, value: Optional[float] = None) -> None:
        try:
            current = float(self.tts_volume_var.get()) if value is None else float(value)
        except (tk.TclError, TypeError, ValueError):
            current = 1.0
        percent = max(0.0, min(2.5, current)) * 100.0
        if self._volume_value_label is not None:
            self._volume_value_label.configure(text=f"{percent:.0f}%")

    def _on_rate_slider_changed(self, value: object) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        numeric = max(0.5, min(2.0, numeric))
        self._update_rate_label(numeric)
        if self._applying_preferences:
            return
        manager = self.tts_manager
        if manager is not None and not math.isclose(numeric, manager.playback_rate(), rel_tol=1e-3):
            manager.set_playback_rate(numeric)
        self._schedule_settings_save()

    def _on_volume_slider_changed(self, value: object) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        numeric = max(0.0, min(2.5, numeric))
        self._update_volume_label(numeric)
        if self._applying_preferences:
            return
        manager = self.tts_manager
        if manager is not None and not math.isclose(numeric, manager.playback_volume(), rel_tol=1e-3):
            manager.set_playback_volume(numeric)
        self._schedule_settings_save()

    def _on_inference_mode_changed(self, _event: object = None) -> None:
        if hasattr(self, "stop_button"):
            current_state = str(self.stop_button.cget("state")).lower()
            if current_state != tk.DISABLED:
                messagebox.showwarning("无法切换", "请先停止当前识别，再切换推理模式。")
                self.inference_mode_var.set(self._last_inference_mode)
                return
        mode = self.inference_mode_var.get()
        self._last_inference_mode = mode
        controller = self._update_active_controller()
        self._update_server_controls()
        if isinstance(controller, RemoteRecognitionController):
            self.post_status("已切换至服务器推理模式。")
        else:
            self.post_status("已切换至本地推理模式。")
            self._schedule_model_preload()
        self._schedule_settings_save()

    def _update_active_controller(self) -> object:
        mode = self.inference_mode_var.get()
        if mode == "服务器推理":
            if self.remote_recognition is None:
                self.remote_recognition = RemoteRecognitionController(self)
            controller: object = self.remote_recognition
        else:
            controller = self.local_recognition
        self.active_controller = controller
        return controller

    def _get_active_controller(self) -> object:
        mode = self.inference_mode_var.get()
        if mode == "服务器推理":
            if self.remote_recognition is None:
                self.remote_recognition = RemoteRecognitionController(self)
            controller: object = self.remote_recognition
        else:
            controller = self.local_recognition
        self.active_controller = controller
        return controller

    def _update_server_controls(self) -> None:
        state = tk.NORMAL if self.inference_mode_var.get() == "服务器推理" else tk.DISABLED
        for entry in (getattr(self, "server_host_entry", None), getattr(self, "server_port_entry", None)):
            if entry is not None:
                entry.configure(state=state)
        button = getattr(self, "remote_test_button", None)
        if button is not None:
            if state == tk.NORMAL and not self._is_remote_test_running():
                button.configure(state=tk.NORMAL)
            else:
                button.configure(state=tk.DISABLED)

    def _is_remote_test_running(self) -> bool:
        thread = getattr(self, "_remote_test_thread", None)
        return thread is not None and thread.is_alive()

    def _test_remote_server(self) -> None:
        if self.inference_mode_var.get() != "服务器推理":
            messagebox.showwarning("无法测试", "请先切换到服务器推理模式。")
            return
        if win32com is None:
            messagebox.showerror("无法生成测试语音", "测试功能依赖 pywin32 (win32com)。")
            return
        if self._is_remote_test_running():
            self.post_status("服务器连通性测试正在进行...")
            return
        host = self.server_host_var.get().strip()
        if not host:
            messagebox.showerror("无法测试", "请填写服务器地址。")
            return
        port_text = self.server_port_var.get().strip()
        try:
            port = int(port_text)
        except ValueError:
            messagebox.showerror("无法测试", "请输入有效的端口号。")
            return
        text = "你好，这是远程识别测试。"
        if hasattr(self, "remote_test_button"):
            self.remote_test_button.configure(state=tk.DISABLED)
        self.post_status("正在发送测试语音到服务器...")
        compare_local = self.inference_mode_var.get() == "本地推理"
        thread = threading.Thread(
            target=self._run_remote_test,
            args=(host, port, text, compare_local),
            name="RemoteServerTest",
            daemon=True,
        )
        self._remote_test_thread = thread
        thread.start()

    def _run_remote_test(self, host: str, port: int, text: str, compare_local: bool) -> None:
        try:
            cache_path = self._ensure_remote_test_wave(text)
            local_result: Optional[str] = None
            if compare_local:
                local_result = self._recognize_wave_locally(cache_path)
            results = self._stream_wave_to_server(cache_path, host, port)
            print(
                "[RemoteTest] partials=", results["partials"],
                "finals=", results["finals"],
                "local=", local_result,
            )
            best_final = self._select_best_segment(results["finals"])
            if not best_final:
                best_final = self._select_best_segment(results["partials"])
            summary = best_final or "无识别结果"

            def on_success(
                server_summary: str = summary,
                local_summary: Optional[str] = local_result,
            ) -> None:
                self.post_status(
                    "服务器测试结果: "
                    + (server_summary if server_summary else "无识别结果"),
                )
                if local_summary is not None:
                    detail = (
                        f"本地识别: {local_summary or '无识别结果'}"
                        f"\n服务器识别: {server_summary or '无识别结果'}"
                    )
                else:
                    detail = f"服务器识别: {server_summary or '无识别结果'}"
                messagebox.showinfo("远程服务器测试", detail)

            self.root.after(0, on_success)
        except Exception as exc:
            error_message = str(exc)

            def on_error(msg: str = error_message) -> None:
                messagebox.showerror("远程服务器测试失败", msg)
                self.post_status(f"服务器测试失败: {msg}")

            self.root.after(0, on_error)
        finally:
            self.root.after(0, self._on_remote_test_finished)

    def _remote_test_wave_path(self) -> str:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "remote-test-sample.wav")

    def _ensure_remote_test_wave(self, text: str) -> str:
        path = self._remote_test_wave_path()
        with self._remote_test_wave_lock:
            if os.path.exists(path) and self._is_valid_remote_test_wave(path):
                return path
            try:
                self._synthesize_tts_to_file(text, path)
            except Exception:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
                raise
            return path

    def _is_valid_remote_test_wave(self, path: str) -> bool:
        try:
            with wave.open(path, "rb") as wav_file:
                return (
                    wav_file.getnchannels() == 1
                    and wav_file.getsampwidth() == 2
                    and wav_file.getframerate() == 16000
                    and wav_file.getnframes() > 0
                )
        except (OSError, wave.Error):
            return False

    def _on_remote_test_finished(self) -> None:
        self._remote_test_thread = None
        self._update_server_controls()

    def _synthesize_tts_to_file(self, text: str, path: str) -> None:
        if win32com is None:
            raise RuntimeError("未找到 pywin32，无法生成测试语音。")
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        audio_format = win32com.client.Dispatch("SAPI.SpAudioFormat")
        audio_format.Type = 0  # 默认输出格式，稍后统一转换
        stream.Format = audio_format
        if os.path.exists(path):
            os.remove(path)
        stream.Open(path, 3, False)
        try:
            voice.AudioOutputStream = stream
            voice.Speak(text)
        finally:
            stream.Close()
            voice.AudioOutputStream = None
        self._ensure_wave_16k_pcm(path)

    def _ensure_wave_16k_pcm(self, path: str) -> None:
        with wave.open(path, "rb") as src:
            params = src.getparams()
            frames = src.readframes(params.nframes)

        sample_width = params.sampwidth
        channels = params.nchannels
        framerate = params.framerate

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1
        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2
        if framerate != 16000:
            frames, _ = audioop.ratecv(frames, sample_width, channels, framerate, 16000, None)
            framerate = 16000

        with wave.open(path, "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sample_width)
            dst.setframerate(framerate)
            dst.writeframes(frames)

    def _recognize_wave_locally(self, path: str) -> Optional[str]:
        if Model is None or KaldiRecognizer is None:
            return None
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.isdir(model_path):
            return None
        try:
            model = Model(model_path)
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.SetWords(True)
        except Exception:
            return None
        try:
            with wave.open(path, "rb") as wav_file:
                if (
                    wav_file.getnchannels() != 1
                    or wav_file.getsampwidth() != 2
                    or wav_file.getframerate() != 16000
                ):
                    return None
                frame_count = wav_file.getnframes()
                chunk_size = max(4000, frame_count // 50 or 4000)
                while True:
                    data = wav_file.readframes(chunk_size)
                    if not data:
                        break
                    recognizer.AcceptWaveform(data)
                final_json = json.loads(recognizer.FinalResult())
                return final_json.get("text", "").strip()
        except Exception:
            return None

    @staticmethod
    def _select_best_segment(segments: List[str]) -> str:
        for text in reversed(segments):
            candidate = text.strip()
            if candidate:
                return candidate
        return ""

    def _stream_wave_to_server(self, path: str, host: str, port: int) -> Dict[str, List[str]]:
        partials: List[str] = []
        finals: List[str] = []
        sock = socket.create_connection((host, port), timeout=5)
        sock.settimeout(1.0)
        reader = sock.makefile("rb")
        writer = sock.makefile("wb")

        def send(payload: Dict[str, object]) -> None:
            data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            writer.write(data)
            writer.flush()

        try:
            with wave.open(path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                if channels != 1 or sample_width != 2 or sample_rate != 16000:
                    raise RuntimeError("测试音频格式不正确，转换失败。")
                send({"type": "hello", "sample_rate": sample_rate, "client": "vtt-debug"})
                ack_raw = reader.readline()
                if not ack_raw:
                    raise RuntimeError("服务器未响应握手。")
                ack = json.loads(ack_raw.decode("utf-8"))
                if ack.get("status") != "ok":
                    raise RuntimeError(ack.get("message", "服务器拒绝连接。"))

                frames_per_chunk = max(int(sample_rate * 0.3), sample_rate // 10)
                while True:
                    chunk = wav_file.readframes(frames_per_chunk)
                    if not chunk:
                        break
                    payload = base64.b64encode(chunk).decode("ascii")
                    send({"type": "audio", "data": payload})

                send({"type": "bye"})
                self._drain_server_stream(reader, partials, finals, wait_seconds=10.0)
        finally:
            try:
                writer.close()
            except Exception:
                pass
            try:
                reader.close()
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        return {"partials": partials, "finals": finals}

    def _drain_server_stream(
        self,
        reader,
        partials: List[str],
        finals: List[str],
        *,
        wait_seconds: float,
    ) -> None:
        timeout = max(wait_seconds, 0.0)
        deadline = time.time() + timeout if timeout > 0 else None
        received = False
        while True:
            try:
                line = reader.readline()
            except (socket.timeout, OSError):
                if deadline is not None and time.time() < deadline:
                    continue
                break
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            msg_type = message.get("type")
            if msg_type == "partial":
                partials.append(message.get("text", ""))
                print("[RemoteTest] recv partial", message)
            elif msg_type == "final":
                text = message.get("text", "")
                if text:
                    finals.append(text)
                print("[RemoteTest] recv final", message)
            elif msg_type == "error":
                raise RuntimeError(message.get("message", "服务器返回错误。"))
            received = True
            if deadline is not None:
                deadline = time.time() + timeout
            if deadline is None:
                break
        if not received and timeout > 0:
            # Nothing new arrived within the timeout; simply return.
            return

    def _select_model_path(self) -> None:
        path = filedialog.askdirectory(title="选择 Vosk 模型目录")
        if path:
            self.model_path_var.set(path)
            self._schedule_model_preload()

    def refresh_devices(self) -> None:
        self._populate_microphones()
        self._populate_engines()
        self._populate_speakers()
        self._populate_voices(refresh_languages=True)

    def _populate_microphones(self) -> None:
        devices: List[str] = []
        self.microphone_map.clear()
        if sd is None:
            devices.append("缺少 sounddevice 库")
            self.microphone_map[devices[0]] = None
        else:
            try:
                for idx, info in enumerate(sd.query_devices()):
                    if info.get("max_input_channels", 0) > 0:
                        name = f"{info.get('name', '设备')} (编号 {idx})"
                        devices.append(name)
                        self.microphone_map[name] = idx
            except Exception as exc:  # pragma: no cover - device enumeration
                devices.append(f"设备枚举失败: {exc}")
                self.microphone_map[devices[0]] = None

        if devices:
            self.mic_combo.configure(values=devices)
            selection = devices[0]
            pending = self._pending_microphone_name
            if pending and pending in self.microphone_map:
                selection = pending
                self._pending_microphone_name = None
            current = self.microphone_var.get()
            if current in self.microphone_map:
                selection = current
            self.mic_combo.current(devices.index(selection))
            self.microphone_var.set(selection)

    def _on_microphone_selected(self, _event: object = None) -> None:
        if self.microphone_var.get() not in self.microphone_map:
            return
        self._schedule_settings_save()

    def _populate_engines(self) -> None:
        combo = getattr(self, "engine_combo", None)
        if combo is None:
            return
        self.engine_map.clear()
        if self.tts_manager is None:
            combo.configure(state=tk.DISABLED, values=["语音合成未初始化"])
            self.engine_var.set("语音合成未初始化")
            return
        engines = self.tts_manager.available_engines()
        if not engines:
            combo.configure(state=tk.DISABLED, values=["未检测到语音引擎"])
            self.engine_var.set("未检测到语音引擎")
            return
        if self._pending_engine_id:
            pending_id = self._pending_engine_id
            if any(engine_id == pending_id for _display, engine_id in engines):
                try:
                    self.tts_manager.set_engine(pending_id)
                except Exception as exc:
                    self.post_status(f"引擎恢复失败: {exc}")
                finally:
                    self._pending_engine_id = None
            else:
                self._pending_engine_id = None
        display_names: List[str] = []
        used: Set[str] = set()
        current_id = self.tts_manager.current_engine()
        current_display: Optional[str] = None
        for display, engine_id in engines:
            base = display or engine_id
            label = base
            suffix = 2
            while label in used:
                label = f"{base} ({suffix})"
                suffix += 1
            used.add(label)
            self.engine_map[label] = engine_id
            display_names.append(label)
            if engine_id == current_id:
                current_display = label
        combo.configure(state="readonly", values=display_names)
        selection = current_display or display_names[0]
        self.engine_var.set(selection)
        combo.set(selection)

    def _populate_speakers(self) -> None:
        self.speaker_map.clear()
        if self.tts_manager is None:
            self.speaker_combo.configure(values=["语音合成未初始化"])
            self.speaker_combo.current(0)
            return
        devices: List[str] = []
        for name, token_id in self.tts_manager.available_outputs():
            devices.append(name)
            self.speaker_map[name] = token_id
        if not devices:
            self.speaker_combo.configure(values=["未检测到朗读输出设备"])
            self.speaker_combo.current(0)
            return
        self.speaker_combo.configure(state="readonly", values=devices)
        selection = devices[0]
        if self._pending_speaker_token_set:
            target = self._pending_speaker_token
            for candidate in devices:
                if self.speaker_map.get(candidate) == target:
                    selection = candidate
                    break
            self._pending_speaker_token_set = False
            self._pending_speaker_token = None
        elif self.speaker_var.get() in self.speaker_map:
            selection = self.speaker_var.get()
        self.speaker_combo.current(devices.index(selection))
        self.speaker_var.set(selection)
        try:
            self.tts_manager.set_output(self.speaker_map.get(selection))
        except Exception as exc:
            self.post_status(f"扬声器切换失败: {exc}")

    def _populate_voices(self, refresh_languages: bool = False) -> None:
        combo = getattr(self, "voice_combo", None)
        if combo is None:
            return
        previous_selection = self.voice_var.get()
        previous_token = self.voice_map.get(previous_selection)
        self.voice_map.clear()
        if self.tts_manager is None:
            combo.configure(state=tk.DISABLED, values=["语音合成未初始化"])
            self.voice_var.set("语音合成未初始化")
            self._disable_language_combo("语音合成未初始化")
            return
        voices = self.tts_manager.available_voices()
        if not voices:
            combo.configure(state=tk.DISABLED, values=["当前语音库不支持音色选择"])
            self.voice_var.set("当前语音库不支持音色选择")
            self._disable_language_combo("语言不可用")
            return
        if refresh_languages or not self.language_map or self.language_var.get() not in self.language_map:
            self._populate_language_options(voices)
        selected_key = self.language_map.get(self.language_var.get(), None)
        filtered: List[VoiceOption] = []
        for option in voices:
            language_value = (option.language or "").strip()
            if selected_key is None:
                filtered.append(option)
            elif selected_key == "":
                if not language_value:
                    filtered.append(option)
            elif selected_key.lower() in language_value.lower():
                filtered.append(option)
        no_match = not filtered and selected_key not in (None, "")
        if no_match:
            self.post_status("所选语言暂无可用音色，已显示默认音色。")
            filtered = []
        options: List[Tuple[str, Optional[str]]] = [("系统默认", None)]
        options.extend((voice.name, voice.token_id) for voice in filtered)
        display_names: List[str] = []
        used: Set[str] = set()
        if self._pending_voice_token_set:
            target_token = self._pending_voice_token
            self._pending_voice_token_set = False
            self._pending_voice_token = None
        else:
            target_token = previous_token
        matched_display: Optional[str] = None
        for idx, (name, token) in enumerate(options):
            base = name or f"音色 {idx + 1}"
            display = base
            suffix = 2
            while display in used:
                display = f"{base} ({suffix})"
                suffix += 1
            used.add(display)
            self.voice_map[display] = token
            display_names.append(display)
            if target_token is None and token is None:
                matched_display = display
            elif target_token is not None and token == target_token:
                matched_display = display
        combo.configure(state="readonly", values=display_names)
        if matched_display and matched_display in self.voice_map:
            self.voice_var.set(matched_display)
        elif display_names:
            self.voice_var.set(display_names[0])
        combo.set(self.voice_var.get())
        token = self.voice_map.get(self.voice_var.get())
        try:
            self.tts_manager.set_voice(token)
        except ValueError as exc:
            messagebox.showerror("音色切换失败", str(exc))
            self.post_status(f"音色切换失败: {exc}")
            self.root.after(0, self._populate_voices)
        finally:
            self._schedule_settings_save()

    def _populate_language_options(self, voices: List[VoiceOption]) -> None:
        combo = getattr(self, "language_combo", None)
        if combo is None:
            return
        previous = self.language_var.get()
        new_map: Dict[str, Optional[str]] = {"全部语言": None}
        values: List[str] = ["全部语言"]
        for option in voices:
            raw = (option.language or "").strip()
            display = raw or "未知语言"
            if display not in new_map:
                new_map[display] = raw if raw else ""
                values.append(display)
        self.language_map = new_map
        combo.configure(state="readonly", values=values)
        target_label: Optional[str] = None
        if self._pending_language_set:
            if self._pending_language_label and self._pending_language_label in new_map:
                expected_value = self._pending_language_value
                actual_value = new_map[self._pending_language_label]
                if expected_value == actual_value:
                    target_label = self._pending_language_label
            if target_label is None and self._pending_language_value is not None:
                for label, value in new_map.items():
                    if value == self._pending_language_value:
                        target_label = label
                        break
            self._pending_language_set = False
            self._pending_language_label = None
            self._pending_language_value = None
        if target_label is None and previous in new_map:
            target_label = previous
        if target_label is None:
            target_label = "全部语言"
        self.language_var.set(target_label)
        combo.set(target_label)

    def _disable_language_combo(self, message: str) -> None:
        combo = getattr(self, "language_combo", None)
        if combo is None:
            return
        self.language_map.clear()
        combo.configure(state=tk.DISABLED, values=[message])
        self.language_var.set(message)
        combo.set(self.language_var.get())

    def start_recognition(self) -> None:
        if self._stopping_recognition:
            LOGGER.info("停止过程尚未完成，暂不开始新的识别")
            return
        if self.tts_manager is None:
            messagebox.showerror("无法启动", "语音合成不可用，请安装 pywin32 或 pyttsx3。")
            return
        device_name = self.microphone_var.get()
        device_index = self.microphone_map.get(device_name)
        if device_index is None:
            messagebox.showerror("无法启动", "请选择有效的麦克风设备。")
            return
        controller = self._get_active_controller()
        self._reset_transcript()

        if isinstance(controller, RemoteRecognitionController):
            host = self.server_host_var.get().strip()
            port_text = self.server_port_var.get().strip()
            try:
                port = int(port_text)
            except ValueError:
                messagebox.showerror("启动失败", "请输入有效的端口号。")
                return
            try:
                controller.start(device_index, host, port)
            except Exception as exc:
                messagebox.showerror("启动失败", str(exc))
                return
            self.post_status("正在连接服务器...")
        else:
            model_path = self.model_path_var.get().strip()
            try:
                controller.start(device_index, model_path)
            except Exception as exc:
                messagebox.showerror("启动失败", str(exc))
                return
            self.post_status("正在加载模型...")

        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self._set_overlay_recording_state("starting")

    def stop_recognition(self) -> None:
        if self._stopping_recognition:
            LOGGER.info("停止请求已在进行，忽略重复操作")
            return
        controller = self._get_active_controller()
        self._stopping_recognition = True
        if hasattr(self, "start_button"):
            self.start_button.configure(state=tk.DISABLED)
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state=tk.DISABLED)
        self.post_status("正在停止识别…")
        worker = threading.Thread(
            target=self._stop_controller_worker,
            args=(controller,),
            name="StopRecognition",
            daemon=True,
        )
        worker.start()

    def _stop_controller_worker(self, controller: object) -> None:
        error: Optional[Exception] = None
        try:
            controller.stop()
        except Exception as exc:  # pragma: no cover - defensive logging
            error = exc
            LOGGER.exception("停止识别过程出现异常")
        finally:
            self.root.after(0, lambda err=error: self._on_stop_complete(err))

    def _on_stop_complete(self, error: Optional[Exception]) -> None:
        self._stopping_recognition = False
        if error is not None:
            message = f"识别停止失败: {error}"
            self.post_status(message)
            try:
                messagebox.showerror("停止失败", message)
            except Exception:
                LOGGER.error("无法弹出停止失败提示: %s", message)
        else:
            self.post_status("识别已停止。")
        if hasattr(self, "start_button"):
            desired_state = tk.DISABLED if self.ptt_enabled_var.get() else tk.NORMAL
            self.start_button.configure(state=desired_state)
        if hasattr(self, "stop_button"):
            self.stop_button.configure(state=tk.DISABLED)
        self._set_overlay_recording_state("idle")

    def _capture_hotkey(self) -> None:
        if pynput_keyboard is None or pynput_mouse is None:
            messagebox.showerror("缺少依赖", "请安装 pynput 以启用热键录制功能。")
            return
        if self._hotkey_capture is not None and self._hotkey_capture.is_running():
            self.post_status("已在监听热键，按 Esc 取消后重试。")
            return

        def on_finish(tokens: List[str]) -> None:
            self.root.after(0, lambda: self._on_hotkey_captured(tokens))

        def on_cancel() -> None:
            self.root.after(0, self._on_hotkey_capture_cancelled)

        try:
            capture = HotkeyCapture(on_finish=on_finish, on_cancel=on_cancel)
        except RuntimeError as exc:
            messagebox.showerror("无法监听热键", str(exc))
            return

        self._hotkey_capture = capture
        self.hotkey_capture_button.configure(state=tk.DISABLED)
        try:
            capture.start()
        except Exception as exc:  # pragma: no cover - defensive
            self._hotkey_capture = None
            self.hotkey_capture_button.configure(state=tk.NORMAL)
            self.post_status(f"热键监听失败: {exc}")
            return
        self.post_status("请按下热键组合，按 Esc 取消。")

    def _on_hotkey_captured(self, tokens: List[str]) -> None:
        self.hotkey_var.set(self._format_hotkey_tokens(tokens))
        self.post_status("热键已捕获，点击应用即可生效。")
        self._finish_hotkey_capture()

    def _on_hotkey_capture_cancelled(self) -> None:
        self.post_status("热键监听已取消。")
        self._finish_hotkey_capture()

    def _finish_hotkey_capture(self) -> None:
        self._hotkey_capture = None
        if hasattr(self, "hotkey_capture_button"):
            self.hotkey_capture_button.configure(state=tk.NORMAL)

    def _cancel_hotkey_capture(self, notify: bool = False) -> None:
        capture = self._hotkey_capture
        if capture is None:
            return
        try:
            capture.stop(notify=notify)
        finally:
            self._hotkey_capture = None
            if hasattr(self, "hotkey_capture_button"):
                self.hotkey_capture_button.configure(state=tk.NORMAL)

    def _apply_hotkey(self) -> None:
        self._cancel_hotkey_capture()
        if self.tts_manager is None:
            messagebox.showerror("无法设置", "语音合成不可用，无法应用热键。")
            return
        text = self.hotkey_var.get().strip()
        if not text:
            self.tts_manager.set_hotkey(None)
            self.post_status("热键已清除。")
            self._schedule_settings_save()
            return
        if pyautogui is None:
            messagebox.showerror("缺少依赖", "请安装 pyautogui 以启用热键功能。")
            return
        try:
            action = self._parse_hotkey(text)
        except ValueError as exc:
            messagebox.showerror("无效热键", str(exc))
            return
        self.tts_manager.set_hotkey(action)
        self.post_status("热键已更新。")
        self._schedule_settings_save()

    def _clear_hotkey(self) -> None:
        self.hotkey_var.set("")
        self._cancel_hotkey_capture()
        if self.tts_manager is not None:
            try:
                self.tts_manager.set_hotkey(None)
            except Exception:
                LOGGER.exception("清除朗读热键失败")
        self.post_status("朗读热键已清除。")
        self._schedule_settings_save()

    def _capture_ptt_hotkey(self) -> None:
        if pynput_keyboard is None or pynput_mouse is None:
            messagebox.showerror("缺少依赖", "请安装 pynput 以启用按键录音设置。")
            return
        if self._ptt_hotkey_capture is not None and self._ptt_hotkey_capture.is_running():
            self.post_status("按键录音按键录制进行中，按 Esc 取消后重试。")
            return

        def on_finish(tokens: List[str]) -> None:
            self.root.after(0, lambda: self._on_ptt_hotkey_captured(tokens))

        def on_cancel() -> None:
            self.root.after(0, self._on_ptt_hotkey_cancelled)

        try:
            capture = HotkeyCapture(on_finish=on_finish, on_cancel=on_cancel)
        except RuntimeError as exc:
            messagebox.showerror("无法监听按键", str(exc))
            return

        self._ptt_hotkey_capture = capture
        if hasattr(self, "ptt_capture_button"):
            self.ptt_capture_button.configure(state=tk.DISABLED)
        try:
            capture.start()
        except Exception as exc:  # pragma: no cover - defensive
            self._ptt_hotkey_capture = None
            if hasattr(self, "ptt_capture_button"):
                self.ptt_capture_button.configure(state=tk.NORMAL)
            self.post_status(f"按键录音监听失败: {exc}")
            return
        self.post_status("请按下按键录音组合，按 Esc 取消。")

    def _finish_ptt_hotkey_capture(self) -> None:
        self._ptt_hotkey_capture = None
        if hasattr(self, "ptt_capture_button"):
            self.ptt_capture_button.configure(state=tk.NORMAL)

    def _on_ptt_hotkey_captured(self, tokens: List[str]) -> None:
        formatted = self._format_hotkey_tokens(tokens)
        self.ptt_hotkey_var.set(formatted)
        self.post_status("按键录音按键已捕获。")
        self._finish_ptt_hotkey_capture()
        self._restart_ptt_listener()
        self._schedule_settings_save()

    def _on_ptt_hotkey_cancelled(self) -> None:
        self.post_status("按键录音监听已取消。")
        self._finish_ptt_hotkey_capture()

    def _cancel_ptt_hotkey_capture(self) -> None:
        capture = self._ptt_hotkey_capture
        if capture is None:
            return
        try:
            capture.stop(notify=True)
        finally:
            self._finish_ptt_hotkey_capture()

    def _clear_ptt_hotkey(self) -> None:
        self._cancel_ptt_hotkey_capture()
        self.ptt_hotkey_var.set("")
        if self.ptt_enabled_var.get():
            self._disable_ptt()
            self.ptt_enabled_var.set(False)
        self.post_status("按键录音按键已清除。")
        self._schedule_settings_save()

    def _capture_emergency_hotkey(self) -> None:
        if pynput_keyboard is None or pynput_mouse is None:
            messagebox.showerror("缺少依赖", "请安装 pynput 以设置急停按钮。")
            return
        if self._emergency_hotkey_capture is not None and self._emergency_hotkey_capture.is_running():
            self.post_status("急停按钮录制进行中，按 Esc 取消后重试。")
            return

        def on_finish(tokens: List[str]) -> None:
            self.root.after(0, lambda: self._on_emergency_hotkey_captured(tokens))

        def on_cancel() -> None:
            self.root.after(0, self._on_emergency_hotkey_cancelled)

        try:
            capture = HotkeyCapture(on_finish=on_finish, on_cancel=on_cancel)
        except RuntimeError as exc:
            messagebox.showerror("无法监听按键", str(exc))
            return

        self._emergency_hotkey_capture = capture
        if hasattr(self, "emergency_capture_button"):
            self.emergency_capture_button.configure(state=tk.DISABLED)
        try:
            capture.start()
        except Exception as exc:  # pragma: no cover - defensive
            self._emergency_hotkey_capture = None
            if hasattr(self, "emergency_capture_button"):
                self.emergency_capture_button.configure(state=tk.NORMAL)
            self.post_status(f"急停按钮监听失败: {exc}")
            return
        self.post_status("请按下急停按钮组合，按 Esc 取消。")

    def _finish_emergency_hotkey_capture(self) -> None:
        self._emergency_hotkey_capture = None
        if hasattr(self, "emergency_capture_button"):
            self.emergency_capture_button.configure(state=tk.NORMAL)

    def _on_emergency_hotkey_captured(self, tokens: List[str]) -> None:
        formatted = self._format_hotkey_tokens(tokens)
        self.emergency_hotkey_var.set(formatted)
        self.post_status("急停按钮已捕获。")
        self._finish_emergency_hotkey_capture()
        self._apply_emergency_hotkey(show_errors=True)
        self._schedule_settings_save()

    def _on_emergency_hotkey_cancelled(self) -> None:
        self.post_status("急停按钮监听已取消。")
        self._finish_emergency_hotkey_capture()

    def _cancel_emergency_hotkey_capture(self) -> None:
        capture = self._emergency_hotkey_capture
        if capture is None:
            return
        try:
            capture.stop(notify=True)
        finally:
            self._finish_emergency_hotkey_capture()

    def _clear_emergency_hotkey(self) -> None:
        self.emergency_hotkey_var.set("")
        self._cancel_emergency_hotkey_capture()
        self._dispose_emergency_listener()
        self._emergency_action = None
        self.post_status("急停按钮已清除。")
        self._schedule_settings_save()

    def _apply_emergency_hotkey(self, *, show_errors: bool) -> None:
        self._dispose_emergency_listener()
        self._emergency_action = None
        text = self.emergency_hotkey_var.get().strip()
        if not text:
            return
        if pynput_keyboard is None:
            message = "请安装 pynput 以启用急停按钮。"
            if show_errors:
                messagebox.showerror("无法启用", message)
            else:
                LOGGER.info("急停按钮未启用: %s", message)
            return
        try:
            action = self._parse_hotkey(text)
        except ValueError as exc:
            detail = str(exc)
            if show_errors:
                messagebox.showerror("急停按钮无效", detail)
            else:
                LOGGER.info("急停按钮未启用: %s", detail)
            return
        if action is None or (not action.keys and not action.mouse_buttons):
            detail = "急停按钮组合不能为空。"
            if show_errors:
                messagebox.showerror("急停按钮无效", detail)
            else:
                LOGGER.info("急停按钮未启用: %s", detail)
            return
        if action.mouse_buttons and pynput_mouse is None:
            detail = "当前缺少 pynput.mouse，无法监听鼠标按键。"
            if show_errors:
                messagebox.showerror("无法启用", detail)
            else:
                LOGGER.info("急停按钮未启用: %s", detail)
            return
        try:
            self._ensure_emergency_listener(action)
            self._emergency_action = action
            if show_errors:
                self.post_status("急停按钮已更新。")
        except Exception as exc:
            detail = str(exc)
            if show_errors:
                messagebox.showerror("急停按钮启用失败", detail)
            else:
                LOGGER.info("急停按钮未启用: %s", detail)

    def _ensure_emergency_listener(self, action: HotkeyAction) -> None:
        self._dispose_emergency_listener()

        def activate() -> None:
            self.root.after(0, self._on_emergency_trigger)

        def deactivate() -> None:
            self.root.after(0, self._on_emergency_trigger)

        listener = PushToTalkListener(action=action, on_activate=activate, on_deactivate=deactivate)
        listener.start()
        self._emergency_listener = listener

    def _dispose_emergency_listener(self) -> None:
        listener = self._emergency_listener
        if listener is not None:
            listener.stop()
        self._emergency_listener = None

    def _on_emergency_trigger(self) -> None:
        self._perform_emergency_stop(source="hotkey")

    def _on_emergency_button(self) -> None:
        self._perform_emergency_stop(source="button")

    def _perform_emergency_stop(self, source: str) -> None:
        if source == "hotkey":
            self.post_status("急停按键已触发，正在停止…")
        elif source == "button":
            self.post_status("急停按钮已触发，正在停止…")
        else:
            self.post_status("急停执行中…")
        if self.tts_manager is not None:
            try:
                self.tts_manager.emergency_stop()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("急停停止朗读失败")
                self.post_status(f"急停停止朗读失败: {exc}")
        if self._is_tts_playing:
            self._is_tts_playing = False
            self.overlay_status_var.set("○ 空闲")
        if self._is_recognition_running():
            try:
                self.stop_recognition()
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("急停停止识别失败")
                messagebox.showerror("急停失败", f"停止识别失败: {exc}")
        else:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
        self._set_overlay_recording_state("idle")
        self._spoken_offset = len(self._final_text)
        self._ptt_active = False
        self._ptt_starting = False
        self._ptt_stop_pending = False
        self.post_status("急停：识别与朗读已停止。")

    def _toggle_ptt(self) -> None:
        try:
            if self.ptt_enabled_var.get():
                if self._is_recognition_running():
                    messagebox.showwarning("无法启用", "请先停止当前识别，再启用按键录音。")
                    self.ptt_enabled_var.set(False)
                    return
                if pynput_keyboard is None:
                    messagebox.showerror("缺少依赖", "请安装 pynput 以启用按键录音功能。")
                    self.ptt_enabled_var.set(False)
                    return
                text = self.ptt_hotkey_var.get().strip()
                if not text:
                    messagebox.showerror("无法启用", "请先设置按键录音按键。")
                    self.ptt_enabled_var.set(False)
                    return
                try:
                    action = self._parse_hotkey(text)
                except ValueError as exc:
                    messagebox.showerror("无法启用", str(exc))
                    self.ptt_enabled_var.set(False)
                    return
                if action is None or (not action.keys and not action.mouse_buttons):
                    messagebox.showerror("无法启用", "按键录音按键组合不能为空。")
                    self.ptt_enabled_var.set(False)
                    return
                try:
                    self._ptt_action = action
                    self._ensure_ptt_listener(action)
                except Exception as exc:
                    messagebox.showerror("无法启用", str(exc))
                    self.ptt_enabled_var.set(False)
                    self._ptt_action = None
                    return
                self.start_button.configure(state=tk.DISABLED)
                self.post_status("按键录音已启用，按下设定的按键即可开始录音。")
            else:
                self._disable_ptt()
                self.post_status("按键录音已关闭。")
        finally:
            self._schedule_settings_save()

    def _restart_ptt_listener(self) -> None:
        if not self.ptt_enabled_var.get():
            return
        text = self.ptt_hotkey_var.get().strip()
        if not text:
            self._disable_ptt()
            self.ptt_enabled_var.set(False)
            return
        try:
            action = self._parse_hotkey(text)
        except ValueError as exc:
            messagebox.showerror("按键录音", f"按键组合无效: {exc}")
            self._disable_ptt()
            self.ptt_enabled_var.set(False)
            return
        if action is None or (not action.keys and not action.mouse_buttons):
            messagebox.showerror("按键录音", "按键录音按键组合不能为空。")
            self._disable_ptt()
            self.ptt_enabled_var.set(False)
            return
        try:
            self._ptt_action = action
            self._ensure_ptt_listener(action)
        except Exception as exc:
            messagebox.showerror("按键录音", str(exc))
            self._disable_ptt()
            self.ptt_enabled_var.set(False)

    def _ensure_ptt_listener(self, action: HotkeyAction) -> None:
        self._dispose_ptt_listener()

        def activate() -> None:
            self.root.after(0, self._ptt_on_trigger_start)

        def deactivate() -> None:
            self.root.after(0, self._ptt_on_trigger_stop)

        listener = PushToTalkListener(action=action, on_activate=activate, on_deactivate=deactivate)
        listener.start()
        self._ptt_listener = listener

    def _dispose_ptt_listener(self) -> None:
        listener = self._ptt_listener
        if listener is not None:
            listener.stop()
        self._ptt_listener = None

    def _disable_ptt(self, *, stop_running: bool = True) -> None:
        self._dispose_ptt_listener()
        self._ptt_action = None
        self._ptt_active = False
        self._ptt_starting = False
        self._ptt_stop_pending = False
        if stop_running and self._is_recognition_running():
            self.stop_recognition()
        if hasattr(self, "start_button"):
            self.start_button.configure(state=tk.NORMAL)

    def _ptt_on_trigger_start(self) -> None:
        if self._ptt_active:
            LOGGER.info("按键录音触发重复按下，忽略")
            return
        LOGGER.info("按键录音按下触发")
        self._ptt_active = True
        self._ptt_stop_pending = False
        self._ptt_begin()

    def _ptt_on_trigger_stop(self) -> None:
        if not self._ptt_active:
            LOGGER.info("按键录音触发松开但未处于激活状态")
            return
        if self._ptt_starting:
            LOGGER.info("按键录音松开时识别仍在启动，标记延迟停止")
            self._ptt_stop_pending = True
            return
        LOGGER.info("按键录音松开，准备停止识别")
        self._stop_ptt_recognition()

    def _ptt_begin(self) -> None:
        if self._ptt_starting:
            LOGGER.info("按键录音启动已在进行，忽略重复请求")
            return
        if self._is_recognition_running():
            LOGGER.info("已有识别在运行，按键录音开始请求被忽略")
            return
        self._ptt_starting = True
        try:
            LOGGER.debug("按键录音：开始识别")
            self.post_status("按键录音：开始识别…")
            self.start_recognition()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("按键录音：启动失败")
            messagebox.showerror("启动失败", f"按键录音启动失败：{exc}")
            self._ptt_active = False
            self._ptt_stop_pending = False
        finally:
            self._ptt_starting = False
            if not self._is_recognition_running():
                self._ptt_active = False
                self._ptt_stop_pending = False
                LOGGER.info("按键录音启动未成功，已重置状态")
            if self._ptt_stop_pending:
                self._ptt_stop_pending = False
                self._stop_ptt_recognition()

    def _stop_ptt_recognition(self) -> None:
        if not self._is_recognition_running():
            self._ptt_active = False
            LOGGER.info("按键录音停止时识别已不在运行")
            return
        LOGGER.info("按键录音停止，发送结束并保持启动按钮禁用")
        self.stop_recognition()
        self._ptt_active = False
        if self.ptt_enabled_var.get() and hasattr(self, "start_button"):
            self.start_button.configure(state=tk.DISABLED)

    def _is_recognition_running(self) -> bool:
        try:
            state = str(self.stop_button.cget("state")).lower()
        except Exception:
            return False
        return state != tk.DISABLED

    def _toggle_overlay(self) -> None:
        if self.show_overlay_var.get():
            self._ensure_overlay()
        else:
            self._destroy_overlay()
        self._schedule_settings_save()

    def _ensure_overlay(self) -> None:
        window = self.overlay_window
        if window is not None:
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if not exists:
                window = None
                self.overlay_window = None
        if window is None:
            window = tk.Toplevel(self.root)
            self.overlay_window = window
            window.withdraw()
            window.configure(bg="#1e1e1e")
            window.overrideredirect(True)
            try:
                window.attributes("-topmost", True)
            except Exception:
                pass
            try:
                window.attributes("-alpha", 0.78)
            except Exception:
                pass
            window.geometry("400x140+200+200")
            window.bind("<Escape>", self._overlay_escape)

            frame = tk.Frame(window, bg="#1e1e1e", padx=12, pady=8)
            frame.pack(fill=tk.BOTH, expand=True)
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=0)
            frame.rowconfigure(1, weight=1)

            status_label = tk.Label(
                frame,
                textvariable=self.overlay_status_var,
                anchor="w",
                bg="#1e1e1e",
                fg="#8ec07c",
                font=("Segoe UI", 9),
            )
            status_label.grid(row=0, column=0, sticky="w")

            record_label = tk.Label(
                frame,
                textvariable=self.overlay_recording_var,
                anchor="e",
                bg="#1e1e1e",
                fg=self._recording_indicator_color,
                font=("Segoe UI", 9),
            )
            record_label.grid(row=0, column=1, sticky="e", padx=(8, 0))
            self.overlay_recording_label = record_label

            text_label = tk.Label(
                frame,
                textvariable=self.overlay_text_var,
                justify=tk.LEFT,
                wraplength=360,
                bg="#1e1e1e",
                fg="#ffffff",
            )
            text_label.grid(row=1, column=0, sticky="nsew")

            for widget in (frame, text_label, status_label, record_label):
                widget.bind("<ButtonPress-1>", self._start_overlay_drag)
                widget.bind("<B1-Motion>", self._on_overlay_drag)

            window.deiconify()

            # Ensure the recording label reflects the current state color.
            record_label.configure(fg=self._recording_indicator_color)

        current_partial = self.partial_var.get()
        self._set_overlay_text(current_partial)

    def _destroy_overlay(self) -> None:
        window = self.overlay_window
        self.overlay_window = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass
        if self.show_overlay_var.get():
            self.show_overlay_var.set(False)

    def _overlay_escape(self, event: Optional[tk.Event] = None) -> None:
        self.show_overlay_var.set(False)
        self._destroy_overlay()
        self._schedule_settings_save()

    def _start_overlay_drag(self, event: tk.Event) -> None:
        self._overlay_drag_offset = (event.x, event.y)

    def _on_overlay_drag(self, event: tk.Event) -> None:
        window = self.overlay_window
        if window is None:
            return
        offset_x, offset_y = self._overlay_drag_offset
        x = event.x_root - offset_x
        y = event.y_root - offset_y
        window.geometry(f"+{x}+{y}")

    def _compose_display_text(self) -> str:
        combined = self._final_text.strip()
        partial = self.partial_var.get().strip()
        if partial:
            combined = f"{combined} {partial}".strip()
        return combined

    def _set_overlay_text(self, text: str, remember: bool = False) -> None:
        sanitized = text.strip()
        if remember:
            self._overlay_last_text = sanitized
        display = sanitized or self._overlay_last_text
        self._update_overlay_text(display)

    def _update_overlay_text(self, text: str) -> None:
        window = self.overlay_window
        if window is None:
            return
        try:
            exists = bool(window.winfo_exists())
        except tk.TclError:
            return
        if not exists:
            return
        display = text if text else "等待识别..."
        self.overlay_text_var.set(display)

    def _set_overlay_recording_state(self, state: str) -> None:
        if state == "active":
            text = "● 录音中"
            color = "#fb4934"
            self._is_recording = True
        elif state == "starting":
            text = "● 准备录音..."
            color = "#fabd2f"
            self._is_recording = False
        else:
            text = "○ 未录音"
            color = "#8ec07c"
            self._is_recording = False
        self.overlay_recording_var.set(text)
        self._recording_indicator_color = color
        label = self.overlay_recording_label
        if label is not None:
            try:
                label.configure(fg=color)
            except tk.TclError:
                self.overlay_recording_label = None

    def _format_hotkey_tokens(self, tokens: List[str]) -> str:
        def beautify(token: str) -> str:
            base = token.lower()
            mouse_display = {
                "mouse_left": "鼠标左键",
                "mouse_right": "鼠标右键",
                "mouse_middle": "鼠标中键",
                "mouse_x1": "鼠标侧键1",
                "mouse_x2": "鼠标侧键2",
            }
            mapping = {
                "ctrl": "Ctrl",
                "alt": "Alt",
                "shift": "Shift",
                "win": "Win",
                "enter": "Enter",
                "space": "Space",
                "tab": "Tab",
                "backspace": "Backspace",
                "delete": "Delete",
                "home": "Home",
                "end": "End",
                "page_up": "Page_Up",
                "page_down": "Page_Down",
                "up": "Up",
                "down": "Down",
                "left": "Left",
                "right": "Right",
                "esc": "Esc",
            }
            if base in mouse_display:
                return mouse_display[base]
            if base in mapping:
                return mapping[base]
            if base.startswith("vk_"):
                return base.upper()
            if len(base) == 1:
                return base.upper()
            return token

        return " + ".join(beautify(token) for token in tokens)

    def _parse_hotkey(self, text: str) -> Optional[HotkeyAction]:
        stripped = text.strip()
        if not stripped:
            return None
        parts = [part.strip() for part in stripped.split("+") if part.strip()]
        if not parts:
            raise ValueError("请提供有效的热键组合，例如 Ctrl+Alt+鼠标左键。")
        mouse_alias_groups = {
            "mouse_left": [
                "mouse_left",
                "mouseleft",
                "leftmouse",
                "mousebutton1",
                "mouse_button1",
                "mouse_button_1",
                "mouse1",
                "button1",
                "mb1",
                "lbutton",
                "left",
                "鼠标左键",
                "鼠标左",
                "左键",
            ],
            "mouse_right": [
                "mouse_right",
                "mouseright",
                "rightmouse",
                "mousebutton2",
                "mouse_button2",
                "mouse_button_2",
                "mouse2",
                "button2",
                "mb2",
                "rbutton",
                "right",
                "鼠标右键",
                "鼠标右",
                "右键",
            ],
            "mouse_middle": [
                "mouse_middle",
                "mousemiddle",
                "middlemouse",
                "mousebutton3",
                "mouse_button3",
                "mouse_button_3",
                "mouse3",
                "button3",
                "mb3",
                "mbutton",
                "middle",
                "鼠标中键",
                "鼠标中",
                "中键",
            ],
            "mouse_x1": [
                "mouse_x1",
                "mousex1",
                "mousebutton4",
                "mouse_button4",
                "mouse_button_4",
                "mouse4",
                "button4",
                "mb4",
                "xbutton1",
                "mouse_xbutton1",
                "mousexbutton1",
                "side1",
                "sidebutton1",
                "side_button1",
                "thumb1",
                "mouse_back",
                "mouseback",
                "back_button",
                "browser_back",
                "back",
                "x1",
                "侧键1",
                "鼠标侧键1",
                "后退键",
                "上一页键",
            ],
            "mouse_x2": [
                "mouse_x2",
                "mousex2",
                "mousebutton5",
                "mouse_button5",
                "mouse_button_5",
                "mouse5",
                "button5",
                "mb5",
                "xbutton2",
                "mouse_xbutton2",
                "mousexbutton2",
                "side2",
                "sidebutton2",
                "side_button2",
                "thumb2",
                "mouse_forward",
                "mouseforward",
                "forward_button",
                "browser_forward",
                "forward",
                "x2",
                "侧键2",
                "鼠标侧键2",
                "前进键",
                "下一页键",
            ],
        }
        mouse_map: Dict[str, str] = {}
        for button, aliases in mouse_alias_groups.items():
            for alias in aliases:
                mouse_map[alias] = button
        keys: List[str] = []
        mouse_buttons: List[str] = []
        for token in parts:
            lowered = token.lower()
            normalized = lowered.replace(" ", "").replace("-", "").replace("_", "")
            mapped = mouse_map.get(normalized)
            if mapped is None:
                mapped = mouse_map.get(lowered)
            if mapped is not None:
                if mapped not in mouse_buttons:
                    mouse_buttons.append(mapped)
                continue
            if lowered.startswith("mouse"):
                raise ValueError(f"不支持的鼠标按键: {token}")
            keys.append(lowered)
        if not keys and not mouse_buttons:
            raise ValueError("热键组合不能为空。")
        return HotkeyAction(keys=keys, mouse_buttons=mouse_buttons)

    def notify_recognition_ready(self) -> None:
        self.root.after(0, self._on_recognition_ready)

    def _on_recognition_ready(self) -> None:
        self.post_status("识别已启动。")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self._set_overlay_recording_state("active")

    def notify_recognition_failed(self, exc: Exception) -> None:
        def handler() -> None:
            messagebox.showerror("启动失败", str(exc))
            self.post_status("识别启动失败。")
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self._set_overlay_recording_state("idle")

        self.root.after(0, handler)

    def handle_partial_result(self, partial: str) -> None:
        self.root.after(0, lambda: self._update_partial(partial))

    def handle_final_result(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self.root.after(0, lambda segment=normalized: self._apply_final_result(segment))

    def _apply_final_result(self, segment: str) -> None:
        cleaned, truncated = self._normalize_final_segment(segment)
        if not cleaned:
            return
        if cleaned == self._last_final_segment:
            return
        if self._final_text.endswith(cleaned):
            self._last_final_segment = cleaned
            self.partial_var.set("")
            self._refresh_text_widget()
            self._set_overlay_text(cleaned, remember=True)
            return
        self._last_final_segment = cleaned
        if self._final_text:
            self._final_text = f"{self._final_text} {cleaned}".strip()
        else:
            self._final_text = cleaned
        self.partial_var.set("")
        self._refresh_text_widget()
        self._set_overlay_text(cleaned, remember=True)
        self._prune_final_history()
        if truncated and not self._truncation_notified:
            self.post_status("识别结果较长，已截断显示以保持流畅。")
            self._truncation_notified = True
        self._speak_new_text()

    def _normalize_final_segment(self, segment: str) -> Tuple[str, bool]:
        cleaned = " ".join(segment.strip().split())
        if not cleaned:
            return "", False
        max_len = 400
        truncated = False
        if len(cleaned) > max_len:
            cleaned = cleaned[:max_len].rstrip()
            truncated = True
        return cleaned, truncated

    def _update_partial(self, partial: str) -> None:
        sanitized = partial.strip()
        if sanitized == self._last_partial_raw and sanitized:
            return
        self._last_partial_raw = sanitized
        if not sanitized:
            self.partial_var.set("")
            self._refresh_text_widget()
            self._set_overlay_text("")
            return
        max_len = 200
        display = sanitized
        if len(display) > max_len:
            display = f"…{display[-max_len:]}"
            if not self._truncation_notified:
                self.post_status("实时转写文本较长，已截断显示以保持流畅。")
                self._truncation_notified = True
        self.partial_var.set(display)
        self._refresh_text_widget()
        self._set_overlay_text(display)

    def _refresh_text_widget(self) -> None:
        combined = self._compose_display_text()
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, combined)
        self.text_widget.configure(state=tk.DISABLED)
        self.text_widget.see(tk.END)

    def _prune_final_history(self) -> None:
        limit = getattr(self, "_final_history_limit", 8000)
        if limit <= 0:
            return
        if len(self._final_text) <= limit:
            return
        drop = len(self._final_text) - limit
        self._final_text = self._final_text[drop:].lstrip()
        self._spoken_offset = max(0, self._spoken_offset - drop)
        if self._spoken_offset > len(self._final_text):
            self._spoken_offset = len(self._final_text)

    def _speak_new_text(self) -> None:
        if self.tts_manager is None:
            return
        if self._spoken_offset >= len(self._final_text):
            return
        new_text = self._final_text[self._spoken_offset :].strip()
        if new_text:
            self._spoken_offset = len(self._final_text)
            self.tts_manager.enqueue(new_text)

    def _on_engine_selected(self, _event: object = None) -> None:
        if self.tts_manager is None:
            return
        selection = self.engine_var.get()
        engine_id = self.engine_map.get(selection)
        if not engine_id:
            return
        try:
            self.tts_manager.set_engine(engine_id)
        except Exception as exc:
            messagebox.showerror("引擎切换失败", str(exc))
            current_id = self.tts_manager.current_engine()
            for display, mapped_id in self.engine_map.items():
                if mapped_id == current_id:
                    self.engine_var.set(display)
                    self.engine_combo.set(display)
                    break
            return
        self.post_status(f"已切换引擎：{selection}")
        self._populate_speakers()
        self._populate_voices(refresh_languages=True)
        self._schedule_settings_save()

    def _on_language_selected(self, _event: object = None) -> None:
        self._populate_voices()
        if self.tts_manager is not None and any(token is not None for token in self.voice_map.values()):
            self.post_status(f"已切换语言：{self.language_var.get()}")
        self._schedule_settings_save()

    def _on_voice_selected(self, _event: object = None) -> None:
        if self.tts_manager is None:
            return
        selection = self.voice_var.get()
        token_id = self.voice_map.get(selection)
        try:
            self.tts_manager.set_voice(token_id)
            self.post_status(f"已切换音色：{selection}")
        except ValueError as exc:
            messagebox.showerror("音色切换失败", str(exc))
            self.post_status(f"音色切换失败: {exc}")
            self.root.after(0, self._populate_voices)
        finally:
            self._schedule_settings_save()

    def _on_speaker_selected(self, event: object) -> None:
        if self.tts_manager is None:
            return
        token_id = self.speaker_map.get(self.speaker_var.get())
        try:
            self.tts_manager.set_output(token_id)
            self.post_status("已切换朗读输出设备。")
        except Exception as exc:
            messagebox.showerror("扬声器切换失败", str(exc))
        finally:
            self._schedule_settings_save()

    def _test_tts(self) -> None:
        if self.tts_manager is None:
            messagebox.showerror("无法测试", "语音合成尚未初始化。")
            return
        self.tts_manager.test()

    def _on_tts_playback_state(self, is_playing: bool) -> None:
        if getattr(self, "root", None) is None:
            return

        def _apply() -> None:
            self._is_tts_playing = is_playing
            label = "● 播放中" if is_playing else "○ 空闲"
            self.overlay_status_var.set(label)

        try:
            self.root.after(0, _apply)
        except Exception:
            # root may be destroyed during shutdown; best effort update.
            _apply()

    def post_status(self, text: str) -> None:
        self.status_var.set(text)

    def _reset_transcript(self) -> None:
        self._final_text = ""
        self._spoken_offset = 0
        self._last_final_segment = ""
        self._last_partial_raw = ""
        self._truncation_notified = False
        self.partial_var.set("")
        self._refresh_text_widget()
        self._set_overlay_text("", remember=True)

    def _on_close(self) -> None:
        LOGGER.info("应用关闭")
        self._cancel_hotkey_capture()
        self._cancel_ptt_hotkey_capture()
        self._cancel_emergency_hotkey_capture()
        self._dispose_emergency_listener()
        self._disable_ptt(stop_running=False)
        self.stop_recognition()
        self._destroy_overlay()
        if self.tts_manager is not None:
            self.tts_manager.shutdown()
        if self._settings_save_job is not None and getattr(self, "root", None) is not None:
            try:
                self.root.after_cancel(self._settings_save_job)
            except Exception:
                pass
            self._settings_save_job = None
        if self._model_preload_job is not None and getattr(self, "root", None) is not None:
            try:
                self.root.after_cancel(self._model_preload_job)
            except Exception:
                pass
            self._model_preload_job = None
        try:
            self._save_settings()
        except Exception:
            LOGGER.exception("关闭时保存设置失败")
        self.root.destroy()


def main() -> None:
    _configure_logging()
    LOGGER.info("应用启动")
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()