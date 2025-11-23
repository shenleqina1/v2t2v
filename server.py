"""简单的 TCP 语音识别服务器，兼容 TranscriptionApp 客户端。

服务器通过 JSON 行接收 base64 编码的 16 位 PCM 音频块，
并将转录结果流式返回给客户端。每个连接的客户端由一个专用线程处理，
因此多个用户可以同时连接。
"""

from __future__ import annotations

import argparse
import contextlib
import audioop
import base64
import json
import logging
import os
import queue
import socket
import sys
import threading
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import wave

try:  # pragma: no cover - 可选依赖项，在运行时验证
	from faster_whisper import WhisperModel  # type: ignore
	from faster_whisper.vad import VadOptions  # type: ignore
except ImportError:  # pragma: no cover - 如果缺少 faster-whisper，快速失败
	WhisperModel = None  # type: ignore
	VadOptions = None  # type: ignore

import numpy as np
import torch

from speaker_verifier import SpeakerVerifier, create_speaker_verifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CT2 = os.path.join(BASE_DIR, "model-ct2")
_DEFAULT_ORIGINAL = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_PATH = _DEFAULT_CT2 if os.path.isdir(_DEFAULT_CT2) else _DEFAULT_ORIGINAL

SILENCE_TIMEOUT = 0.6  # 低能量持续时间（秒）后提交最终结果

LOGGER = logging.getLogger("whisper_server")
ENERGY_THRESHOLD = 350  # 根据麦克风增益调整
MIN_SPEECH_DURATION = 0.35  # 超过阈值的最短语音持续时间（秒）


def _patch_torchaudio_load() -> None:
	try:
		import torchaudio  # pylint: disable=import-error
	except ImportError:
		return
	if getattr(_patch_torchaudio_load, "_patched", False):
		return
	original_load = torchaudio.load

	def _safe_load(uri, *args, **kwargs):
		try:
			return original_load(uri, *args, **kwargs)
		except RuntimeError as exc:
			if isinstance(uri, str) and uri.lower().endswith(".wav"):
				return _load_wav(uri)
			raise exc

	torchaudio.load = _safe_load  # type: ignore[attr-defined]
	_patch_torchaudio_load._patched = True


def _load_wav(path: str) -> Tuple[torch.Tensor, int]:
	with contextlib.closing(wave.open(path, "rb")) as wf:
		sampwidth = wf.getsampwidth()
		if sampwidth != 2:
			raise RuntimeError("声纹参考音频仅支持 16-bit PCM WAV。")
		num_channels = wf.getnchannels()
		num_frames = wf.getnframes()
		sample_rate = wf.getframerate()
		pcm_bytes = wf.readframes(num_frames)

	audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
	if num_channels > 1:
		audio = audio.reshape(-1, num_channels).mean(axis=1)
	waveform = torch.from_numpy(audio).unsqueeze(0)
	return waveform, sample_rate


class StreamingWhisperSession:
	"""使用共享的 Whisper 模型在 GPU 上处理流式音频。"""

	def __init__(
		self,
		model: Any,
		sample_rate: int,
		language: str = "zh",
		speaker_verifier: Optional[SpeakerVerifier] = None,
		vad_options: Optional[Any] = None,
	) -> None:
		if sample_rate != 16000:
			raise ValueError("Whisper 仅支持 16kHz 单声道 PCM。请调整客户端采样率。")
		self._model = model
		self._language = language
		self._speaker_verifier = speaker_verifier
		self._vad_options = vad_options
		self._audio_queue: "queue.Queue[Tuple[str, Optional[np.ndarray]]]" = queue.Queue(maxsize=48)
		self._result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
		self._thread = threading.Thread(target=self._run, name="whisper-session", daemon=True)
		self._started = False
		self._stop_requested = threading.Event()
		self._buffer: list[np.ndarray] = []
		self._total_samples = 0
		self._samples_since_partial = 0
		self._min_partial_samples = int(sample_rate * 0.8)
		self._partial_step_samples = int(sample_rate * 0.4)
		self._last_partial_text = ""
		self._sample_rate = sample_rate

	def start(self) -> None:
		if not self._started:
			self._thread.start()
			self._started = True

	def push_audio(self, pcm_bytes: bytes) -> None:
		if not pcm_bytes:
			return
		samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
		try:
			self._audio_queue.put(("audio", samples), timeout=5.0)
		except queue.Full:
			LOGGER.warning("音频队列已满，丢弃一帧音频以避免阻塞。")

	def finalize(self) -> None:
		try:
			self._audio_queue.put(("finalize", None), timeout=1.0)
		except queue.Full:
			LOGGER.warning("音频队列繁忙，无法立即提交最终识别。")

	def close(self) -> None:
		self._stop_requested.set()
		try:
			self._audio_queue.put_nowait(("stop", None))
		except queue.Full:
			self._audio_queue.put(("stop", None))
		if self._started:
			self._thread.join(timeout=5.0)

	def drain_results(self) -> list[Dict[str, Any]]:
		items: list[Dict[str, Any]] = []
		while True:
			try:
				items.append(self._result_queue.get_nowait())
			except queue.Empty:
				break
		return items

	def _run(self) -> None:
		try:
			while not self._stop_requested.is_set():
				command, payload = self._audio_queue.get()
				if command == "audio" and payload is not None:
					self._buffer.append(payload)
					samples = payload.shape[0]
					self._total_samples += samples
					self._samples_since_partial += samples
					self._emit_partial_if_ready()
				elif command == "finalize":
					self._emit_final()
				elif command == "stop":
					break
		except Exception as exc:  # pragma: no cover - 防御性代码
			LOGGER.exception("Whisper 识别线程异常")
			self._result_queue.put({"type": "error", "message": f"识别失败: {exc}"})
		finally:
			self._result_queue.put({"type": "complete"})

	def _combine_buffer(self) -> Optional[np.ndarray]:
		if not self._buffer:
			return None
		return np.concatenate(self._buffer)

	def _transcribe_buffer(self) -> str:
		audio = self._combine_buffer()
		if audio is None or audio.size == 0:
			return ""
		return self._transcribe_audio(audio)

	def _emit_partial_if_ready(self) -> None:
		if self._total_samples < self._min_partial_samples:
			return
		if self._samples_since_partial < self._partial_step_samples:
			return
		text = self._transcribe_buffer()
		if text and text != self._last_partial_text:
			self._result_queue.put({"type": "partial", "text": text})
			self._last_partial_text = text
		self._samples_since_partial = 0

	def _emit_final(self) -> None:
		audio = self._combine_buffer()
		if audio is None or audio.size == 0:
			self._reset_buffer()
			return
		text = self._transcribe_audio(audio)
		if not text:
			self._reset_buffer()
			return
		score: Optional[float] = None
		if self._speaker_verifier is not None:
			torch_audio = torch.from_numpy(audio).float().unsqueeze(0)
			result = self._speaker_verifier.verify(torch_audio, self._sample_rate)
			score = result.score
			if not result.accepted:
				self._result_queue.put({"type": "speaker_reject", "score": score})
				self._result_queue.put({"type": "partial", "text": ""})
				self._reset_buffer()
				return
		payload: Dict[str, Any] = {"type": "final", "text": text}
		if score is not None:
			payload["score"] = score
		self._result_queue.put(payload)
		self._result_queue.put({"type": "partial", "text": ""})
		self._reset_buffer()

	def _reset_buffer(self) -> None:
		self._buffer.clear()
		self._total_samples = 0
		self._samples_since_partial = 0
		self._last_partial_text = ""

	def _transcribe_audio(self, audio: np.ndarray) -> str:
		transcribe_kwargs: Dict[str, Any] = {
			"language": self._language,
			"beam_size": 1,
			"condition_on_previous_text": False,
			"temperature": 0.0,
		}
		if self._vad_options is not None:
			transcribe_kwargs["vad_filter"] = True
			transcribe_kwargs["vad_parameters"] = self._vad_options
		else:
			transcribe_kwargs["vad_filter"] = False
		segments, _ = self._model.transcribe(audio, **transcribe_kwargs)
		return "".join(seg.text.strip() for seg in segments).strip()


def _send_json(writer, payload: Dict[str, Any]) -> None:
	data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
	writer.write(data)
	writer.flush()


def _handle_client(
	conn: socket.socket,
	address: tuple[str, int],
	model: Any,
	speaker_verifier: Optional[SpeakerVerifier],
	vad_options: Optional[Any],
) -> None:
	client_tag = f"{address[0]}:{address[1]}"
	LOGGER.info("客户端连接: %s", client_tag)
	conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	reader = conn.makefile("rb")
	writer = conn.makefile("wb")

	def send(payload: Dict[str, Any]) -> None:
		try:
			_send_json(writer, payload)
		except Exception:
			pass

	def send_status(status: str, message: Optional[str] = None) -> None:
		payload: Dict[str, Any] = {"status": status}
		if message:
			payload["message"] = message
		send(payload)

	session: Optional[StreamingWhisperSession] = None

	try:
		try:
			raw = reader.readline()
		except (OSError, ConnectionError) as exc:
			LOGGER.warning("%s 握手读取失败: %s", client_tag, exc)
			return
		if not raw:
			LOGGER.warning("%s 握手前即断开", client_tag)
			return
		try:
			hello = json.loads(raw.decode("utf-8"))
		except json.JSONDecodeError:
			LOGGER.warning("%s 握手消息无法解析: %r", client_tag, raw)
			send_status("error", "无法解析握手消息。")
			return
		if hello.get("type") != "hello":
			LOGGER.warning("%s 握手消息类型错误: %s", client_tag, hello.get("type"))
			send_status("error", "握手消息格式不正确。")
			return
		try:
			sample_rate = int(hello.get("sample_rate", 0))
		except (TypeError, ValueError):
			LOGGER.warning("%s 握手 sample_rate 无效: %s", client_tag, hello.get("sample_rate"))
			send_status("error", "sample_rate 必须为整数。")
			return
		if sample_rate <= 0:
			LOGGER.warning("%s 握手 sample_rate 非法: %s", client_tag, sample_rate)
			send_status("error", "sample_rate 必须为正整数。")
			return
		if sample_rate != 16000:
			LOGGER.warning("%s 握手 sample_rate=%s 不受支持", client_tag, sample_rate)
			send_status("error", "当前服务器仅支持 16000Hz PCM 音频。")
			return

		session = StreamingWhisperSession(
			model,
			sample_rate,
			speaker_verifier=speaker_verifier,
			vad_options=vad_options,
		)
		session.start()
		send_status("ok")
		LOGGER.info("%s 握手完成，采样率=%s", client_tag, sample_rate)

		energy_window: Deque[Tuple[int, float]] = deque(maxlen=40)
		stream_failed = False
		silence_duration = 0.0
		speech_active = False

		while True:
			try:
				raw = reader.readline()
			except (OSError, ConnectionError) as exc:
				LOGGER.info("%s 读取音频数据时连接终止: %s", client_tag, exc)
				break
			if not raw:
				LOGGER.info("%s 客户端发送结束", client_tag)
				break
			try:
				message = json.loads(raw.decode("utf-8"))
			except json.JSONDecodeError:
				LOGGER.warning("%s 收到无法解析的消息: %r", client_tag, raw[:60])
				send({"type": "error", "message": "收到无法解析的消息。"})
				continue

			msg_type = message.get("type")
			if msg_type == "audio":
				data_field = message.get("data")
				if not isinstance(data_field, str):
					LOGGER.warning("%s 音频数据不是字符串", client_tag)
					send({"type": "error", "message": "音频数据格式错误。"})
					continue
				try:
					chunk = base64.b64decode(data_field)
				except (base64.binascii.Error, ValueError):
					LOGGER.warning("%s 音频数据解码失败", client_tag)
					send({"type": "error", "message": "无法解码音频数据。"})
					continue
				if len(chunk) % 2:
					continue
				try:
					rms = audioop.rms(chunk, 2)
				except audioop.error:
					continue
				chunk_duration = len(chunk) / 2 / sample_rate
				energy_window.append((rms, chunk_duration))
				session.push_audio(chunk)
				if rms >= ENERGY_THRESHOLD:
					speech_active = True
					silence_duration = 0.0
				else:
					silence_duration += chunk_duration
					if speech_active and silence_duration >= SILENCE_TIMEOUT:
						session.finalize()
						speech_active = False
						silence_duration = 0.0

				for event in session.drain_results():
					event_type = event.get("type")
					if event_type == "partial":
						send({"type": "partial", "text": event.get("text", "")})
					elif event_type == "final":
						text = event.get("text", "").strip()
						if not text:
							continue
						window_duration = sum(d for _, d in energy_window)
						energy_above = sum(d for e, d in energy_window if e >= ENERGY_THRESHOLD)
						avg_energy = (
							sum(e * d for e, d in energy_window) / window_duration
							if window_duration > 0
							else 0.0
						)
						if energy_above < MIN_SPEECH_DURATION and len(text) <= 2:
							LOGGER.debug(
								"%s 低能量识别被忽略: %s (avg_energy=%.1f)",
								client_tag,
								text,
								avg_energy,
							)
							continue
						LOGGER.info("%s 最终识别: %s", client_tag, text)
						payload = {"type": "final", "text": text}
						if "score" in event:
							payload["score"] = event["score"]
						send(payload)
						energy_window.clear()
					elif event_type == "speaker_reject":
						score = event.get("score")
						LOGGER.info("%s 说话人不匹配，score=%.3f", client_tag, score or -1.0)
						send({"type": "error", "message": "说话人与目标声纹不匹配。"})
					elif event_type == "error":
						send(event)
						stream_failed = True
						break
					elif event_type == "complete":
						pass

				if stream_failed:
					break
			elif msg_type == "bye":
				LOGGER.info("%s 收到 bye", client_tag)
				if session is not None:
					session.finalize()
				break
			else:
				LOGGER.warning("%s 收到未知消息类型: %s", client_tag, msg_type)
				send({"type": "error", "message": "收到未知消息类型。"})

		if session is not None:
			try:
				session.finalize()
			except Exception:
				pass
			session.close()
			for event in session.drain_results():
				if event.get("type") == "final" and event.get("text"):
					payload = {"type": "final", "text": event["text"]}
					if "score" in event:
						payload["score"] = event["score"]
					send(payload)
				elif event.get("type") == "partial":
					send({"type": "partial", "text": event.get("text", "")})
				elif event.get("type") == "speaker_reject":
					score = event.get("score")
					send({"type": "error", "message": "说话人与目标声纹不匹配。", "score": score})
			session = None
	finally:
		if session is not None:
			try:
				session.close()
			except Exception:
				pass
		try:
			writer.close()
		except Exception:
			pass
		try:
			reader.close()
		except Exception:
			pass
		try:
			conn.close()
		except Exception:
			pass
		LOGGER.info("客户端断开: %s", client_tag)


def _serve(
	host: str,
	port: int,
	model_id: str,
	device: str,
	compute_type: str,
	speaker_ref: Optional[str],
	speaker_device: str,
	speaker_threshold: float,
	enable_vad: bool,
	vad_threshold: float,
	vad_min_speech: float,
	vad_min_silence: float,
	vad_pad: float,
) -> None:
	if WhisperModel is None:
		LOGGER.error("缺少 faster-whisper 库，请执行 pip install faster-whisper")
		sys.exit(1)

	LOGGER.info("正在加载 Whisper 模型: %s", model_id)
	try:
		model = WhisperModel(model_id, device=device, compute_type=compute_type)
	except Exception as exc:
		LOGGER.error("加载 Whisper 模型失败: %s", exc)
		sys.exit(1)
	LOGGER.info("模型加载完成，启动服务器 %s:%s", host, port)
	_patch_torchaudio_load()
	vad_options: Optional[Any] = None
	if enable_vad:
		if VadOptions is None:
			LOGGER.warning("当前 faster-whisper 版本不支持内置 VAD，已自动禁用。")
		else:
			vad_options = VadOptions(
				threshold=vad_threshold,
				min_speech_duration_ms=max(1, int(vad_min_speech * 1000)),
				min_silence_duration_ms=max(1, int(vad_min_silence * 1000)),
				speech_pad_ms=max(0, int(vad_pad * 1000)),
			)
	verifier = create_speaker_verifier(
		speaker_ref,
		device=speaker_device,
		threshold=speaker_threshold,
	)

	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_socket.bind((host, port))
	server_socket.listen()

	try:
		while True:
			conn, address = server_socket.accept()
			thread = threading.Thread(
				target=_handle_client,
				args=(conn, address, model, verifier, vad_options),
				daemon=True,
			)
			thread.start()
	except KeyboardInterrupt:
		LOGGER.info("收到中断，正在关闭服务器...")
	finally:
		server_socket.close()


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Whisper 语音识别服务器")
	parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
	parser.add_argument("--port", type=int, default=8765, help="监听端口，默认 8765")
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL_PATH,
		help=(
			"Whisper 模型 ID 或本地目录，默认使用当前目录下的 model 子目录。"
		),
	)
	parser.add_argument("--device", default="cuda", help="运行设备，示例: cuda、cpu")
	parser.add_argument(
		"--compute-type",
		default="int8_float16",  # 修改默认值为 int8_float16
		help="推理精度，常用 float16 / int8_float16。",
	)
	parser.add_argument(
		"--speaker-ref",
		help="目标说话人参考音频路径，用于声纹过滤。",
	)
	parser.add_argument(
		"--speaker-threshold",
		type=float,
		default=0.75,
		help="声纹相似度阈值，越高越严格 ( 默认 0.75)。",
	)
	parser.add_argument(
		"--speaker-device",
		default="cuda",
		help="声纹模型运行设备，示例: cuda、cpu。",
	)
	parser.add_argument(
		"--vad-disable",
		action="store_true",
		help="禁用 faster-whisper 内置 VAD 声学检测。",
	)
	parser.add_argument(
		"--vad-threshold",
		type=float,
		default=0.5,
		help="VAD 触发阈值，范围 0-1，值越大越严格 (默认 0.5)。",
	)
	parser.add_argument(
		"--vad-min-speech",
		type=float,
		default=0.3,
		help="VAD 判定语音的最短持续时间（秒），默认 0.3。",
	)
	parser.add_argument(
		"--vad-min-silence",
		type=float,
		default=0.5,
		help="VAD 判定静音的最短持续时间（秒），默认 0.5。",
	)
	parser.add_argument(
		"--vad-pad",
		type=float,
		default=0.2,
		help="VAD 输出包含的前后留白（秒），默认 0.2。",
	)
	return parser.parse_args()


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",
	)
	args = _parse_args()
	_serve(
		args.host,
		args.port,
		args.model,
		args.device,
		args.compute_type,
		args.speaker_ref,
		args.speaker_device,
		args.speaker_threshold,
		not args.vad_disable,
		args.vad_threshold,
		args.vad_min_speech,
		args.vad_min_silence,
		args.vad_pad,
	)


if __name__ == "__main__":
	main()
