from FlagEmbedding import BGEM3FlagModel
from faster_whisper import WhisperModel; 

whisper_mode = WhisperModel('turbo')
model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=False,
                       batch_size=8,
                       trust_remote_code= True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
