
import os
import asyncio
import io
import re
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import asyncio
import time
import numpy as np
import soundfile as sf

from indextts.infer_vllm import IndexTTS

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS(model_dir=args.model_dir, gpu_memory_utilization=args.gpu_memory_utilization)

    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))

        for speaker, audio_paths in speaker_dict.items():
            audio_paths_ = []
            for audio_path in audio_paths:
                audio_paths_.append(os.path.join(cur_dir, audio_path))
            tts.registry_speaker(speaker, audio_paths_)
    yield
    # Clean up the ML models and release the resources
    # ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 添加CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        global tts
        if tts is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "TTS model not initialized"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "message": "Service is running",
                "timestamp": time.time()
            }
        )
    except Exception as ex:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(ex)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]
        seed = data.get("seed", 8)

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


def parse_llm_response(response_text):
    """
    解析 LLM 回复，提取情感标签和语音内容。

    Args:
        response_text (str): 大模型的原始回复文本，例如 "【happy】今天真不错！"

    Returns:
        dict: 包含 'tag' (str) 和 'content' (str) 的字典。
    """

    # 1. 预处理：去除首尾空白字符
    text = response_text.strip()

    # 2. 定义正则模式
    # ^             : 匹配字符串开头
    # [\[【]        : 兼容中文 '【' 或 英文 '['
    # (.*?)         : 捕获组1（标签），非贪婪匹配，提取标签内容
    # [\]】]        : 兼容中文 '】' 或 英文 ']'
    # \s* : 允许标签和内容之间有任意空格
    # (.*)          : 捕获组2（内容），匹配剩余所有文本
    # re.DOTALL     : 让 '.' 也能匹配换行符（防止回答包含换行时截断）
    pattern = r"^[\[【](.*?)[\]】]\s*(.*)$"

    match = re.match(pattern, text, re.DOTALL)

    if match:
        # 提取并清洗数据
        raw_tag = match.group(1).strip()
        content = match.group(2).strip()

        # 3. 安全校验：确保提取的标签在你的允许列表中
        # 如果模型输出了 'excited' 这种不在列表里的词，强制转为 'normal'
        valid_tags = {
            'normal', 'angry', 'disgusted', 'fear', 'happy',
            'sad', 'shock', 'loudly', 'quietly', 'whisper'
        }

        # 将标签转为小写比较，增加鲁棒性
        tag = raw_tag.lower() if raw_tag.lower() in valid_tags else 'normal'

        return {
            "tag": tag,
            "content": content
        }
    else:
        # 4. 兜底策略 (Fallback)
        # 如果正则未匹配（例如模型忘了加标签），默认使用 normal
        return {
            "tag": "normal",
            "content": text
        }


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )



@app.get("/audio/voices")
async def tts_voices():
    """ additional function to provide the list of available voices, in the form of JSON """
    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))
        return speaker_dict
    else:
        return []


@app.post("/audio/speech", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_openai(request: Request):
    """ OpenAI competible API, see: https://api.openai.com/v1/audio/speech """
    try:
        data = await request.json()
        text = data["input"]
        character = data["voice"]
        llm_result = parse_llm_response(text)
        text = llm_result.get("content")
        style = llm_result.get("tag")
        if character == "orange":
            character = f"{character}_{style}"
        # model param is omitted
        _model = data["model"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    args = parser.parse_args()

    uvicorn.run(app=app, host=args.host, port=args.port)
