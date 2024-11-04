from vosk import Model, KaldiRecognizer
import wave
import json
import uuid
from pathlib import Path
import io

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import re

class DeMater:
    def __init__(self, model_path= "models/vosk-model-small-ru-0.22", beep_filename="data/censor-beep-2-2.wav"):
        self.model = Model(model_path=model_path)

        txt = Path('words.txt').read_text(encoding='utf-8')
        self.target_word_list_default = txt.replace('\n', ',')
        self.user_data = {}

        self.initWisperModel()
        #print(self.target_word_list_default)

    def initWisperModel(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-small"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipeWisperModel = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        #D:\venv\gpu\Scripts\activate.bat

    def get_user_data_or_new(self, session_id):
        if session_id not in self.user_data:
            self.user_data[session_id] = {}

        return self.user_data[session_id]

    def get_target_word_list_or_default(self, session_id):
        custom = ""
        if session_id in self.user_data:
            custom = self.user_data[session_id]["target_word_list_custom"]
            
        return custom if custom != "" else self.target_word_list_default

    def get_beep_audio(self, framerate, session_id=None):

        beep_filenames = {
            "8000": "data/censor-beep-2-8000.wav",
            "16000": "data/censor-beep-2-16000.wav",
            "32000": "data/censor-beep-2-32000.wav",
            "44100": "data/censor-beep-2-44100.wav",
            "48000": "data/censor-beep-2-48000.wav",
        }

        if session_id in self.user_data:
            session_data = self.user_data[session_id]
            if "beep_data" in session_data:
                beep_filenames = session_data["beep_data"]

        beep_filename = "data/censor-beep-2-2.wav"
        if str(framerate) in beep_filenames:
            beep_filename = beep_filenames[str(framerate)]
            

        beep_wf = wave.open(beep_filename, 'rb')
        beep_data = beep_wf.readframes(beep_wf.getnframes())
        print(f'beep_wf.getparams={beep_wf.getparams()}, beep_data.len={len(beep_data)}')

        return beep_data

    def get_text_from_audio__whisper(self, input_file):

        input_file.seek(0)
        data = []
        wave_params = {}
        with wave.open(input_file, 'rb') as input_wf:
            wave_params = input_wf.getparams()
            #wav.setparams((wave_params.nchannels, wave_params.sampwidth, wave_params.framerate, 0, wave_params.comptype, wave_params.compname))
            data = input_wf.readframes(wave_params.nframes)
            #data = bytearray(data)
            data_s16 = np.frombuffer(data, dtype=np.int16, count=len(data)//2, offset=0)
            data = data_s16 * 0.5**15

        sample = {
            "array": data,
            "sampling_rate": wave_params.framerate
        }

        result = self.pipeWisperModel(sample, return_timestamps="word", chunk_length_s=30, generate_kwargs={"language": "russian"})

        print(f'result-whisper={result}')

        """
        result = {
            "text": result["text"],
            "result": result["chunks"]
        }
        """

        return result

    def get_text_from_audio(self, input_file="test4.wav"):
        wf = wave.open(input_file, 'rb')
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        print(f'input_file.getparams= {wf.getparams()}')
        print(wf.getframerate())

        while True:
            data = wf.readframes(16000) #4000)
            if len(data) == 0:
                break
            
            rec.AcceptWaveform(data)
            """if rec.AcceptWaveform(data):
                print(rec.Result())
            else:
                print(rec.PartialResult())
                """

        return rec.FinalResult()

    def replace_text(self, input_text, detected_word_list):
        words = [detected_word if type(detected_word) == str else detected_word["word"] for detected_word in detected_word_list]
        words = list(dict.fromkeys(words)) if len(words) > 0 else []
        
        # telegram.error.BadRequest: Can't parse entities: character '-' is reserved and must be escaped with the preceding '\'
        input_text = re.sub(r"([-_*\[\]()~`>#+=|{}.!])", r"\\\g<1>", input_text)
        input_tokens = input_text.split()

        input_tokens_replaced = []
        for input_token in input_tokens:
            input_token_test = input_token.lower()
            if input_token_test in words:
                input_tokens_replaced.append("||" + input_token + "||")
            else:
                word_matched = ""
                for word in words:
                    if word in input_token_test and len(word) > len(word_matched):
                        word_matched = word
                
                if word_matched != "":
                    startIndex = input_token_test.find(word_matched)
                    endIndex = startIndex + len(word_matched)
                    word_to_replace = input_token[startIndex:endIndex]
                    if ((input_token[startIndex-1:startIndex] == ''\
                            or not input_token[startIndex-1:startIndex].isalpha())\
                        and (input_token[endIndex:endIndex+1] == ''\
                            or not input_token[endIndex:endIndex+1].isalpha())):
                        input_token = input_token.replace(word_to_replace, "||" + word_to_replace + "||")
                
                input_tokens_replaced.append(input_token)
        
        return ' '.join(input_tokens_replaced)
                    

    def replace_audio(self, input_file, detected_word_list, session_id=None, padding=0.1):
        #out_file = "tmp-" + str(uuid.uuid4()) + ".wav"# + input_file
        out_file = io.BytesIO()
        out_file.seek(0)

        with wave.open(out_file, 'wb') as wav:

            data = []
            input_file.seek(0)
            with wave.open(input_file, 'rb') as input_wf:
                wave_params = input_wf.getparams()
                wav.setparams((wave_params.nchannels, wave_params.sampwidth, wave_params.framerate, 0, wave_params.comptype, wave_params.compname))

                data = input_wf.readframes(wave_params.nframes)
                print(len(data))
                print(len(bytearray(data)))
                data = bytearray(data)

                for detected_word in detected_word_list:
                    start = detected_word["start"]
                    end = detected_word["end"]

                    start = min(start, start + padding)
                    end = max(end, end - padding)
                    end = min(end, start+0.7)

                    startIndex = int(max(min(start * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0)) // 2 * 2
                    endIndex = int(max(min(end * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0)) // 2 * 2
                    replace_count = int(endIndex - startIndex) // 2 * 2
                    endIndex2 = startIndex + replace_count
                    beep_data = self.get_beep_audio(wave_params.framerate, session_id)

                    data[startIndex:endIndex2] =  beep_data[:replace_count]

                    print(f'replace_count={replace_count}, start={start}, startIndex={startIndex}, end={end}, endIndex={endIndex}, wave_params.framerate={wave_params.framerate},wave_params.nframes={wave_params.nframes}, data.len={len(data)}')
                
            wav.writeframes(data)

        out_file.seek(0)
        return out_file

    def process(self, input_file="test4.wav", target_words=None, session_id=None):
        result = self.get_text_from_audio(input_file=input_file)
        #print(result)
        result = json.loads(result)

        target_words = target_words if target_words is not None else self.target_word_list_default
        target_word_list = [word for word in target_words.split(',') if word.strip() != ""]

        detected_word_list = []
        if "result" in result:
            detected_word_list = [item for item in result["result"] if item["word"] in target_word_list]
        detected_word_list_count = len(detected_word_list)
        
        resultWhisper = self.get_text_from_audio__whisper(input_file)
        detected_word_list2_count = 0
        if "chunks" in resultWhisper:
            for item in resultWhisper["chunks"]:
                word = item["text"].strip().replace("?", "").replace("!", "").replace(",", "").replace(".", "").lower()
                if word in target_word_list:
                    detected_word = {"word": word, "start": item["timestamp"][0], "end": item["timestamp"][1]}
                    detected_word_list.append(detected_word)
                    detected_word_list2_count = detected_word_list2_count + 1

        out_file = self.replace_audio(input_file, detected_word_list, session_id)
        out_text = self.replace_text(result["text"], detected_word_list)

        out_text_whisper = self.replace_text(resultWhisper["text"], detected_word_list)

        print(f'out_text={out_text}, detected_word_list={detected_word_list}')

        return {
            "out_file": out_file,
            "text": out_text,
            "detected_word_list_count": detected_word_list_count,
            "text_whisper": out_text_whisper,
            "detected_word_list2_count": detected_word_list2_count
        }