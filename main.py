import tempfile
import os
import json
from typing import Annotated, Literal, Optional
import logging

import base64

import pydantic
from fastapi import FastAPI, File, BackgroundTasks, Depends, HTTPException, Query, Body, UploadFile, Form
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.callbacks import get_openai_callback
from pydantic import BaseModel, Field, model_validator
import requests
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace

from src.agent import Chain
from src.config import load_config
from src.retriever import setup_retriever

config = load_config()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Customize the log format
)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

model_to_chain = {
    key: Chain(model=key, system_prompt=system_prompt, retriever=retriever).build()
    for key, system_prompt, retriever in [
        ("llama-3.1-8b", open(config.system_prompt_path).read(), setup_retriever()),
        ("little-antipov-llama-3.1-7b:latest", open(config.system_prompt_path).read(), None),
    ]
}

#FastAPI application setup
app = FastAPI(
    title="Little Antipov chatbot",
    description="""Talk to Little Antipov""",
    version="0.0.1",
)
header_scheme = APIKeyHeader(name="api_key")


#CORS (Cross-Origin Resource Sharing) middleware, allows the API to be accessed from different domains or origins. 

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Checking health of application, returns “OK” JSON response

@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:   path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


class Message(BaseModel):
    """User message"""
    model: str = Field(description="Which model to use", default="little-antipov-llama-3.1-7b:latest")
    text: str = Field(description="Text of the message", examples=["Привет! Как дела?"])
    thread_id: str = Field(description="Dialog ID", examples=["1"])
    system: Optional[str] = Field(description="System prompt", default=None)


class Answer(BaseModel):
    """Model answer"""
    thread_id: str = Field(description="Dialog ID")
    text: str | None = Field(description="Text of the answer", default=None)


def _get_chain_input(message: Message, image: UploadFile | None = None, doc_path: str | None = None):
    if image is not None:
        image = image.file.read()
        image = base64.b64encode(image).decode('utf-8')
    
    msg_content = [
        {"type": "text", "text": message.text},
    ]
    if image is not None:
        msg_content.append({
            "type": "image_url", 
            "image_url": {
                "url":  f"data:image/jpeg;base64,{image}"
            }
        })
    msg = HumanMessage(
        content=msg_content
    )
    result = {
        "messages": [msg]
    }
    if doc_path is not None:
        loader = UnstructuredLoader(
            doc_path,
        )
        docs = loader.load()
        result["documents"] = docs
    if message.system is not None:
        result["system"] = message.system
    return result


async def send_message_to_webhook(
    message: Message, 
    webhook: str, 
    image: UploadFile | None = None, 
    doc_path: str | None = None, 
    stream=False
):
    # TODO: depersonalizer callback and fix stream mode to wait if "<" appears
    chain = model_to_chain[message.model]
    if not stream:
        msg = await chain.ainvoke(
            input=_get_chain_input(message=message, image=image, doc_path=doc_path), 
            config={"configurable": {"thread_id": message.thread_id}},
        )
        msg = msg["messages"][-1].content
        requests.post(webhook, json=Answer(
            text=msg, 
            thread_id=message.thread_id,
        ).model_dump())
    else:
        async for msg, metadata in chain.astream(input=_get_chain_input(message=message, image=image, doc_path=doc_path),
            config={"configurable": {"thread_id": message.thread_id}},
            stream_mode="messages",
        ):
            requests.post(
                webhook, 
                json=Answer(text=msg.content, thread_id=message.thread_id).model_dump()
            )



@app.post("/message", description="Send next message to GPT")
async def send_message(
    background_tasks: BackgroundTasks,
    message: str = Form(...),  # Use Form to handle JSON as a string
    stream: Annotated[bool, Query(description="Send answer by parts")]= False, 
    webhook: Annotated[str | None, Query(description="Webhook, where model answer will be sent", example="http://example.com/accept_message")] = None,
    # token: str = Depends(header_scheme),
    image: Optional[UploadFile] = None,
    doc: Optional[UploadFile] = None,
) -> Answer | None:
    try:
        message_data = json.loads(message)  # Parse the JSON string
        message = Message.model_validate(message_data, strict=True)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in message field")
    except pydantic.ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid message: {e}")
    
    if doc is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + doc.filename.split(".")[-1]) as temp_file:
            temp_file.write(doc.file.read())
            doc_path = temp_file.name
    else:
        doc_path = None

    if False:
    # if token not in config["valid_api_keys"]:
        raise HTTPException(status_code=403, detail="Token is invalid")
    if webhook is None:
        chain = model_to_chain[message.model]
        msg = await chain.ainvoke(
            input=_get_chain_input(message=message, image=image, doc_path=doc_path), 
            config={"configurable": {"thread_id": message.thread_id}},
        )
        msg = msg["messages"][-1].content
        return Answer(
            text=msg,
            thread_id=message.thread_id
        )
    else:
        background_tasks.add_task(
            send_message_to_webhook,
            message,
            webhook,
            image,
            doc_path,
            stream
        )

