import os
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
import torch
import torchaudio
import subprocess
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# FastAPI 인스턴스 생성
app = FastAPI()
# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print("Loading model...")
config = XttsConfig()
config.load_json("../workspace/run/training/GPTTrain/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="../workspace/run/training/GPTTrain/", use_deepspeed=False)
model.to(device)

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["../workspace/wavs/output-0001.wav"])


@app.get("/tts")
def get_wav(query):
    try:
        print("Inference...")
        input_text = query
        lang = "ja"  # 언어 설정
        temperature = 0.7  # temperature 값 설정

        # 모델의 최대 포지션 임베딩 길이 확인
        max_position_embeddings = 100  # 예시값, 실제 모델 설정에 맞게 조정

        # Encode the input text with the language specified
        input_ids = model.tokenizer.encode(input_text, lang)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        # 패딩 토큰 ID 정의 (일반적으로 0으로 설정)
        pad_token_id = 0

        print(f"Initial input_ids size: {input_ids.size(1)}")
        max_position_embeddings = input_ids.size(1) + 35
        # Ensure the input_ids are the correct length
        if input_ids.size(1) < max_position_embeddings:
            padding_length = max_position_embeddings - input_ids.size(1)
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), 'constant', pad_token_id)
        elif input_ids.size(1) > max_position_embeddings:
            input_ids = input_ids[:, :max_position_embeddings]

        print(f"Adjusted input_ids size: {input_ids.size(1)}")
        attention_mask = torch.ones(input_ids.shape, device=device)

        # Ensure the attention_mask matches the input_ids length
        if attention_mask.size(1) > max_position_embeddings:
            attention_mask = attention_mask[:, :max_position_embeddings]

        print(f"Final input_ids size: {input_ids.size(1)}")
        print(f"Attention mask size: {attention_mask.size(1)}")

        # Generate audio with the specified temperature
        out = model.inference(
            text=input_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,  # Add temperature parameter
            attention_mask=attention_mask  # Provide attention mask
        )

        torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0).cpu(), 24000)
        command = ["ffmpeg", "-y", "-i", "xtts.wav", "-ar",
                    "44100", "-ac", "1", "-acodec", "pcm_s16le", "output_wav.wav"]
        subprocess.run(command, check=True)
        return FileResponse('./output_wav.wav', media_type='audio/wav')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)