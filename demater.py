from vosk import Model, KaldiRecognizer
import wave
import json
import uuid
from pathlib import Path
import io

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

class DeMater:
    def __init__(self, model_path= "models\\vosk-model-small-ru-0.22", beep_filename="data\\censor-beep-2-2.wav"):
        self.model = Model(model_path=model_path)
        self.beep_wf = wave.open(beep_filename, 'rb')
        self.beep_data = self.beep_wf.readframes(self.beep_wf.getnframes())
        
        print(self.beep_wf.getparams())
        print(len(self.beep_data))
        #print(self.beep_data)

        txt = Path('words.txt').read_text(encoding='utf-8')
        self.target_word_list_default = txt.replace('\n', ',')
        self.target_word_list_custom = {}

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

    def get_target_word_list_or_default(self, session_id):
        custom = ""
        if session_id in self.target_word_list_custom:
            custom = self.target_word_list_custom[session_id]
            
        return custom if custom != "" else self.target_word_list_default

    def get_beep_audio(self, framerate):

        beep_filenames = {
            "8000": "data\\censor-beep-2-8000.wav",
            "16000": "data\\censor-beep-2-16000.wav",
            "32000": "data\\censor-beep-2-32000.wav",
            "44100": "data\\censor-beep-2-44100.wav",
            "48000": "data\\censor-beep-2-48000.wav",
        }

        beep_filename = "data\\censor-beep-2-2.wav"
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

        result = self.pipeWisperModel(sample, return_timestamps="word", generate_kwargs={"language": "russian"})

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

    def mask_text(self, input_text, mask_word_list):
        input_text = ' '.join(["||" + word + "||" if word in mask_word_list else word for word in input_text.split(' ')])

        return input_text

    def replace_text(self, input_text, detected_word_list):
        words = [detected_word["word"] for detected_word in detected_word_list]
        words = list(dict.fromkeys(words)) if len(words) > 0 else []
        return self.mask_text(input_text, words)

    def replace_audio(self, input_file, detected_word_list, padding=0.2):
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

                    startIndex = int(max(min(start * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0)) // 2 * 2
                    endIndex = int(max(min(end * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0)) // 2 * 2
                    replace_count = int((endIndex - startIndex) * (1 - padding)) // 2 * 2
                    endIndex2 = startIndex + replace_count
                    beep_data = self.get_beep_audio(wave_params.framerate)

                    data[startIndex:endIndex2] =  beep_data[:replace_count]

                    print(f'replace_count={replace_count}, start={start}, startIndex={startIndex}, end={end}, endIndex={endIndex}, wave_params.framerate={wave_params.framerate},wave_params.nframes={wave_params.nframes}, data.len={len(data)}')
                
            wav.writeframes(data)

        out_file.seek(0)
        return out_file

    def process(self, input_file="test4.wav", target_words=None):
        result = self.get_text_from_audio(input_file=input_file)
        #print(result)
        result = json.loads(result)

        target_words = target_words if target_words is not None else self.target_word_list_default
        target_word_list = [word for word in target_words.split(',') if word.strip() != ""]

        detected_word_list = []
        if "result" in result:
            detected_word_list = [item for item in result["result"] if item["word"] in target_word_list]
        
        resultWhisper = self.get_text_from_audio__whisper(input_file)
        if "chunks" in resultWhisper:
            for item in resultWhisper["chunks"]:
                word = item["text"].strip().replace("!", "").replace(",", "").replace(",", "").lower()
                if word in target_word_list:
                    detected_word = {"word": word, "start": item["timestamp"][0], "end": item["timestamp"][1]}
                    detected_word_list.append(detected_word)


        out_file = self.replace_audio(input_file, detected_word_list)
        out_text = self.replace_text(result["text"], detected_word_list)
        out_text = out_text if out_text != "" else "<no text>"
        # telegram.error.BadRequest: Can't parse entities: character '-' is reserved and must be escaped with the preceding '\'
        out_text = out_text.replace("-", "\\-")
        # telegram.error.BadRequest: Can't parse entities: character '>' is reserved and must be escaped with the preceding '\'
        out_text = out_text.replace(">", "\\>").replace("<", "\\<").replace(".", "\\.")
        
        print(f'out_text={out_text}, detected_word_list={detected_word_list}')

        return {
            "out_file": out_file,
            "text": out_text,
            "detected_word_list_count": len(detected_word_list)
        }