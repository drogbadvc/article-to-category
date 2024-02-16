import asyncio
import httpx
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline
from bs4 import BeautifulSoup
from cachetools import cached, TTLCache
import trafilatura
import logging

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Global error handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred."},
    )


# Create a cache for titles. Each item will expire after 600 seconds.
title_cache = TTLCache(maxsize=100, ttl=600)


class Item(BaseModel):
    url_input: Optional[List[str]] = Field(default=None)
    text_input: Optional[List[str]] = Field(default=None)
    option: str
    word_list: List[str]


classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# Shared HTTPX client instance
client = httpx.AsyncClient()

logging.basicConfig(level=logging.INFO)


# @cached(title_cache)
async def get_title(url):
    if url in title_cache:
        return title_cache[url]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    try:
        logging.info(f"Attempt to retrieve the title for the URL : {url}")
        response = await client.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string.strip()
        title_cache[url] = title
        return title
    except Exception as e:
        logging.error(f"Error retrieving title for URL {url}: {e}")
        return None


async def get_url_title(urls):
    titles = {}
    tasks = [asyncio.create_task(get_title(url)) for url in urls]
    results = await asyncio.gather(*tasks)
    for url, title in zip(urls, results):
        titles[url] = title
    return titles


async def extract_text(urls):
    results = {}
    for label, url in urls.items():
        downloaded = await asyncio.get_event_loop().run_in_executor(None, trafilatura.fetch_url, url)
        if downloaded:
            result = trafilatura.extract(downloaded, no_fallback=True, include_comments=False,
                                         include_tables=False, favor_precision=True)
            results[label] = result
        else:
            results[label] = None
    return results


@app.post("/classify/")
async def classify(item: Item):
    if item.url_input is None and item.text_input is None:
        return {"error": "You must provide either url_input or text_input"}

    if item.option == "Title" and item.url_input is None:
        return {"error": "To use the 'Title' option, you must provide url_input"}

    sequences_to_classify = {}
    if item.url_input:
        sequences_to_classify = await get_url_title(item.url_input) if item.option == "Title" else await extract_text(
            item.url_input)

    if item.text_input:
        sequences_to_classify.update({f"text{i}": text for i, text in enumerate(item.text_input)})

    result = {}
    for url, sequence in sequences_to_classify.items():
        if sequence:
            output = classifier(sequence, item.word_list, multi_label=False)
            result[url] = output
        else:
            result[url] = "Unable to extract sequence to classify"
    return result
