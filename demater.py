from vosk import Model, KaldiRecognizer
import wave
import json
import uuid
from pathlib import Path
import io

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
        #print(self.target_word_list_default)

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
            "44100": "data\\censor-beep-2-48000.wav",
            "48000": "data\\censor-beep-2-48000.wav",
        }

        beep_filename = "data\\censor-beep-2-2.wav"
        if str(framerate) in beep_filenames:
            beep_filename = beep_filenames[str(framerate)]
            

        beep_wf = wave.open(beep_filename, 'rb')
        beep_data = beep_wf.readframes(beep_wf.getnframes())

        return beep_data

    def get_text_from_audio(self, input_file="test4.wav"):
        wf = wave.open(input_file, 'rb')
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        print(wf.getparams())
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
        words = []
        for word in mask_word_list:
            if(word not in words):
                input_text = input_text.replace(word, "||" + word + "||")
                words.append(word)

        return input_text

    def replace_text(self, input_text, detected_word_list):
        words = [detected_word["word"] for detected_word in detected_word_list]
        return self.mask_text(input_text, words)

    def replace_audio(self, input_file, detected_word_list, padding=0.5):
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

                    startIndex = int(max(min(start * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0))
                    endIndex = int(max(min(end * wave_params.framerate * wave_params.sampwidth, wave_params.nframes * wave_params.sampwidth ), 0))
                    replace_count = int((endIndex - startIndex) * (1 - padding))
                    endIndex2 = startIndex + replace_count
                    beep_data = self.get_beep_audio(wave_params.framerate)
                    data[startIndex:endIndex2] =  beep_data[:replace_count]

                    print(f'replace_count={replace_count}, start={start}, startIndex={startIndex}, end={end}, endIndex={endIndex}, wave_params.framerate={wave_params.framerate},wave_params.nframes={wave_params.nframes}')
                
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

        print(detected_word_list)

        out_file = self.replace_audio(input_file, detected_word_list)
        out_text = self.replace_text(result["text"], detected_word_list)

        return {
            "out_file": out_file,
            "text": out_text
        }