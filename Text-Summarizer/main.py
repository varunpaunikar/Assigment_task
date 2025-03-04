import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import List, AsyncGenerator

import yaml
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from groq import Groq
from langdetect import detect


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        return "en"


app = FastAPI()

CONFIG_FILE = "config.yml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)

groq_key = config['api_keys']['groq']

MAX_FILE_SIZE = 50 * 1024 * 1024


def validate_txt_file(file: UploadFile) -> None:
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only TXT files are allowed."
        )
    
    try:
        content = file.file.read()
        file.file.seek(0)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024}MB"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class Processing:
    def __init__(self, key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            self.client = Groq(api_key=key) 
            self.model = model
        except Exception as e:
            raise ValueError(f"Error initializing Groq client: {e}")

    def _chunk_text(self, context: str, chunk_size: int = 7000) -> List[str]:
        return [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]

    async def generate_summaries(self, context: str, system_prompt: str, user_prompt: str, sleep_time: int = 2) -> AsyncGenerator[str, None]:
        chunks = self._chunk_text(context)
        language = detect_language(context)

        for idx, chunk in enumerate(chunks):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt.format(chunk=chunk, language=language)},
                        {"role": "user", "content": user_prompt}
                    ]
                )

                yield completion.choices[0].message.content

                await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"Error generating summary for chunk: {e}")

    async def merge_summary(self, summaries: List[str], system_prompt: str, user_prompt: str, sleep_time: int = 2) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.format(language=detect_language(summaries[0]))},
                    {"role": "user", "content": user_prompt.format(summaries=' '.join(summaries))}
                ],
                temperature=0.2
            )

            await asyncio.sleep(sleep_time)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error merging summaries: {e}")
            return ''


class Summarize(Processing):
    def __init__(self, key: str, model: str = "llama-3.3-70b-versatile"):
        super().__init__(key, model)

    async def summarize(self, context: str) -> AsyncGenerator[str, None]:
        system_prompt = """You are a text summarizer tasked with condensing the provided text into a cohesive and concise summary.  
        Ensure the summary retains key points and maintains logical flow. Ensure the summary is written in the {language} language as the provided text. The text is: {chunk}"""

        user_prompt = "Summarize the following text and generate a compact summary."
        
        async for summary in self.generate_summaries(context, system_prompt, user_prompt):
            yield summary

    async def summarize_txt(self, input_text: str) -> AsyncGenerator[str, None]:
        async for summary in self.summarize(input_text):
            yield summary


@app.post("/txt_summarizer")
async def upload_txt(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            validate_txt_file(file)
            
            temp_dir_path = Path(temp_dir)
            input_path = temp_dir_path / file.filename
            
            try:
                with open(input_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
            except Exception as e:
                raise HTTPException(status_code=500, detail="Error saving uploaded file")
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            except Exception as e:
                raise HTTPException(status_code=500, detail="Error reading TXT content.")

            summarizer = Summarize(groq_key)  
            
            async def summary_generator():
                try:
                    async for summary in summarizer.summarize_txt(input_text=extracted_text):
                        yield summary + "\n\n"
                except Exception as e:
                    raise HTTPException(status_code=500, detail="Error generating summary.")

            return StreamingResponse(summary_generator(), media_type="text/plain")

        except HTTPException as e:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers=2)