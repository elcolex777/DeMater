import io
import json
import os
import re
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from vosk import KaldiRecognizer, Model


class DeMater:
    def __init__(
        self,
        vosk_model_path: str = "models/vosk-model-small-ru-0.22",
        whisper_model_id: str = "openai/whisper-small",
        faster_whisper_model_size: str = "small",
        # Флаги включения/отключения методов обработки
        enable_vosk: bool = True,
        enable_faster_whisper: bool = True,
        enable_whisper: bool = False,
    ):
        self.enable_vosk = enable_vosk
        self.enable_faster_whisper = enable_faster_whisper
        self.enable_whisper = enable_whisper

        # Инициализация словаря матерных слов
        words_path = Path("words.txt")
        if words_path.exists():
            txt = words_path.read_text(encoding="utf-8")
            self.target_word_list_default = txt.replace("\n", ",")
        else:
            self.target_word_list_default = ""

        self.user_data: Dict[str, Any] = {}

        # 1. Vosk
        self.vosk_model: Optional[Model] = None
        if self.enable_vosk:
            if os.path.exists(vosk_model_path):
                self.vosk_model = Model(model_path=vosk_model_path)
            else:
                print(f"[DeMater] Vosk model not found at {vosk_model_path}. Disabling Vosk.")
                self.enable_vosk = False

        # 2. Faster-Whisper
        self.faster_whisper_model = None
        if self.enable_faster_whisper:
            self._init_faster_whisper(faster_whisper_model_size)

        # 3. Transformers Whisper
        self.pipe_whisper = None
        if self.enable_whisper:
            self._init_whisper_transformers(whisper_model_id)

    def _init_faster_whisper(self, model_size: str):
        try:
            from faster_whisper import WhisperModel

            # int8 на CPU даёт минимальное потребление RAM (до 400-600МБ) и высокий FPS
            self.faster_whisper_model = WhisperModel(
                model_size_or_path=model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count() or 2,
            )
        except Exception as e:
            print(f"[DeMater] Error loading faster-whisper: {e}")
            self.enable_faster_whisper = False

    def _init_whisper_transformers(self, model_id: str):
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(device)
            processor = AutoProcessor.from_pretrained(model_id)

            self.pipe_whisper = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )
        except Exception as e:
            print(f"[DeMater] Error loading Transformers Whisper: {e}")
            self.enable_whisper = False

    def get_user_data_or_new(self, session_id: str) -> dict:
        session_id_str = str(session_id)
        if session_id_str not in self.user_data:
            self.user_data[session_id_str] = {}
        return self.user_data[session_id_str]

    def get_target_word_list_or_default(self, session_id: str) -> str:
        session_id_str = str(session_id)
        custom = ""
        if session_id_str in self.user_data:
            custom = self.user_data[session_id_str].get("target_word_list_custom", "")
        return custom if custom != "" else self.target_word_list_default

    def get_beep_audio(self, framerate: int, session_id: Optional[str] = None) -> bytes:
        beep_filenames = {
            "8000": "data/censor-beep-2-8000.wav",
            "16000": "data/censor-beep-2-16000.wav",
            "32000": "data/censor-beep-2-32000.wav",
            "44100": "data/censor-beep-2-44100.wav",
            "48000": "data/censor-beep-2-48000.wav",
        }

        if session_id and str(session_id) in self.user_data:
            session_data = self.user_data[str(session_id)]
            if "beep_data" in session_data:
                beep_filenames = session_data["beep_data"]

        beep_filename = beep_filenames.get(str(framerate), "data/censor-beep-2-16000.wav")

        if not os.path.exists(beep_filename):
            beep_filename = "data/censor-beep-2-2.wav"

        if os.path.exists(beep_filename):
            with wave.open(beep_filename, "rb") as beep_wf:
                return beep_wf.readframes(beep_wf.getnframes())

        # Fallback: генерация тишины, если аудиофайла нет
        return b"\x00" * int(framerate * 2)

    # --- ДВИЖОК 1: Vosk ---
    def get_text_from_audio__vosk(self, input_file: io.BytesIO) -> dict:
        if not self.enable_vosk or not self.vosk_model:
            return {"text": "", "result": []}

        input_file.seek(0)
        with wave.open(input_file, "rb") as wf:
            rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
            rec.SetWords(True)
            rec.SetPartialWords(False)

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            final_res = json.loads(rec.FinalResult())
            return {
                "text": final_res.get("text", ""),
                "result": final_res.get("result", []),
            }

    # --- ДВИЖОК 2: Faster-Whisper ---
    def get_text_from_audio__faster_whisper(self, input_file: io.BytesIO) -> dict:
        if not self.enable_faster_whisper or not self.faster_whisper_model:
            return {"text": "", "chunks": []}

        input_file.seek(0)
        with wave.open(input_file, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            data_s16 = np.frombuffer(frames, dtype=np.int16)
            audio_data = data_s16.astype(np.float32) / 32768.0

        segments, _ = self.faster_whisper_model.transcribe(
            audio_data,
            language="ru",
            word_timestamps=True,
            vad_filter=True,
        )

        full_text_list = []
        chunks = []
        for segment in segments:
            full_text_list.append(segment.text.strip())
            if segment.words:
                for word in segment.words:
                    chunks.append(
                        {
                            "text": word.word,
                            "timestamp": (word.start, word.end),
                        }
                    )

        return {
            "text": " ".join(full_text_list),
            "chunks": chunks,
        }

    # --- ДВИЖОК 3: Transformers Whisper ---
    def get_text_from_audio__whisper(self, input_file: io.BytesIO) -> dict:
        if not self.enable_whisper or not self.pipe_whisper:
            return {"text": "", "chunks": []}

        input_file.seek(0)
        with wave.open(input_file, "rb") as input_wf:
            wave_params = input_wf.getparams()
            frames = input_wf.readframes(wave_params.nframes)
            data_s16 = np.frombuffer(frames, dtype=np.int16)
            audio_data = data_s16.astype(np.float32) / 32768.0

        sample = {
            "array": audio_data,
            "sampling_rate": wave_params.framerate,
        }

        result = self.pipe_whisper(
            sample,
            return_timestamps="word",
            chunk_length_s=30,
            generate_kwargs={"language": "russian"},
        )
        return {
            "text": result.get("text", ""),
            "chunks": result.get("chunks", []),
        }

    def replace_text(self, input_text: str, detected_word_list: list) -> str:
        words = [
            w if isinstance(w, str) else w["word"]
            for w in detected_word_list
            if w
        ]
        words = list(dict.fromkeys(words))
        if not words or not input_text:
            return input_text

        input_text = re.sub(r"([-_*\[\]()~`>#+=|{}.!])", r"\\\g<1>", input_text)
        input_tokens = input_text.split()

        input_tokens_replaced = []
        for input_token in input_tokens:
            input_token_test = input_token.lower()
            if input_token_test in words:
                input_tokens_replaced.append(f"||{input_token}||")
            else:
                word_matched = ""
                for word in words:
                    if word in input_token_test and len(word) > len(word_matched):
                        word_matched = word

                if word_matched != "":
                    startIndex = input_token_test.find(word_matched)
                    endIndex = startIndex + len(word_matched)
                    word_to_replace = input_token[startIndex:endIndex]
                    left_char = input_token[startIndex - 1 : startIndex]
                    right_char = input_token[endIndex : endIndex + 1]

                    if (left_char == "" or not left_char.isalpha()) and (
                        right_char == "" or not right_char.isalpha()
                    ):
                        input_token = input_token.replace(
                            word_to_replace, f"||{word_to_replace}||"
                        )

                input_tokens_replaced.append(input_token)

        return " ".join(input_tokens_replaced)

    def replace_audio(
        self,
        input_file: io.BytesIO,
        detected_word_list: list,
        session_id: Optional[str] = None,
        padding: float = 0.05,
    ) -> io.BytesIO:
        out_file = io.BytesIO()
        input_file.seek(0)

        with wave.open(input_file, "rb") as input_wf:
            wave_params = input_wf.getparams()
            raw_frames = bytearray(input_wf.readframes(wave_params.nframes))

            with wave.open(out_file, "wb") as wav:
                wav.setparams(wave_params)

                beep_data = self.get_beep_audio(wave_params.framerate, session_id)

                for detected_word in detected_word_list:
                    start = max(0.0, detected_word["start"] - padding)
                    end = detected_word["end"] + padding

                    start_idx = int(start * wave_params.framerate) * wave_params.sampwidth
                    end_idx = int(end * wave_params.framerate) * wave_params.sampwidth

                    start_idx = max(0, min(start_idx, len(raw_frames))) // 2 * 2
                    end_idx = max(0, min(end_idx, len(raw_frames))) // 2 * 2

                    replace_len = end_idx - start_idx
                    if replace_len > 0:
                        # Повторяем бип, если интервал длиннее сэмпла
                        filler = (beep_data * ((replace_len // len(beep_data)) + 1))[:replace_len]
                        raw_frames[start_idx:end_idx] = filler

                wav.writeframes(raw_frames)

        out_file.seek(0)
        return out_file

    def _extract_words_from_chunks(self, chunks: list, target_word_set: set) -> list:
        detected = []
        for item in chunks:
            raw_word = (
                item["text"]
                .strip()
                .replace("?", "")
                .replace("!", "")
                .replace(",", "")
                .replace(".", "")
                .lower()
            )
            if raw_word in target_word_set:
                ts = item.get("timestamp", (0.0, 0.0))
                start = ts[0] if ts[0] is not None else 0.0
                end = ts[1] if ts[1] is not None else start + 0.3
                detected.append({"word": raw_word, "start": start, "end": end})
        return detected

    def process(
        self,
        input_file: io.BytesIO,
        target_words: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        target_words_raw = target_words if target_words is not None else self.target_word_list_default
        target_word_set = {
            w.strip().lower() for w in target_words_raw.split(",") if w.strip()
        }

        all_detected_words = []
        methods_data = {}

        # 1. Vosk
        if self.enable_vosk:
            vosk_res = self.get_text_from_audio__vosk(input_file)
            vosk_detected = [
                item for item in vosk_res.get("result", [])
                if item.get("word", "").lower() in target_word_set
            ]
            all_detected_words.extend(vosk_detected)
            methods_data["vosk"] = {
                "name": "Vosk",
                "text": vosk_res.get("text", ""),
                "masked_text": self.replace_text(vosk_res.get("text", ""), vosk_detected),
                "count": len(vosk_detected),
            }

        # 2. Faster-Whisper
        if self.enable_faster_whisper:
            fw_res = self.get_text_from_audio__faster_whisper(input_file)
            fw_detected = self._extract_words_from_chunks(fw_res.get("chunks", []), target_word_set)
            all_detected_words.extend(fw_detected)
            methods_data["faster_whisper"] = {
                "name": "Whisper (Faster)",
                "text": fw_res.get("text", ""),
                "masked_text": self.replace_text(fw_res.get("text", ""), fw_detected),
                "count": len(fw_detected),
            }

        # 3. Transformers Whisper
        if self.enable_whisper:
            whisper_res = self.get_text_from_audio__whisper(input_file)
            whisper_detected = self._extract_words_from_chunks(whisper_res.get("chunks", []), target_word_set)
            all_detected_words.extend(whisper_detected)
            methods_data["whisper"] = {
                "name": "Whisper (Original)",
                "text": whisper_res.get("text", ""),
                "masked_text": self.replace_text(whisper_res.get("text", ""), whisper_detected),
                "count": len(whisper_detected),
            }

        # Удаление дубликатов по временным отсечкам для чистого наложения звукового сигнала
        unique_detected = []
        for word_info in all_detected_words:
            is_dup = False
            for u in unique_detected:
                # Если слова перекрываются во времени больше чем на 100мс
                if abs(word_info["start"] - u["start"]) < 0.15:
                    is_dup = True
                    break
            if not is_dup:
                unique_detected.append(word_info)

        out_file = self.replace_audio(input_file, unique_detected, session_id)

        return {
            "out_file": out_file,
            "methods": methods_data,
            "total_detected_words": len(unique_detected),
            # Поля для обратной совместимости
            "text": methods_data.get("vosk", {}).get("masked_text", ""),
            "detected_word_list_count": methods_data.get("vosk", {}).get("count", 0),
            "text_whisper": methods_data.get("faster_whisper", {}).get("masked_text", ""),
            "detected_word_list2_count": methods_data.get("faster_whisper", {}).get("count", 0),
        }