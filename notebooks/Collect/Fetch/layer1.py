# sends requests through curl_cffi

import asyncio
import contextlib
import time
from uuid import uuid4

from curl_cffi import requests
from fastapi import FastAPI, HTTPException, Request

sessions = {}
sessions_lock = asyncio.Lock()

async def remove_expired_sessions():
    try:
        while True:
            current_time = time.time()
            async with sessions_lock:
                expired_sessions = [
                    sid
                    for sid, (sess, last_access) in sessions.items()
                    if current_time - last_access > SESSION_TIMEOUT
                ]
                for sid in expired_sessions:
                    await sessions[sid][0].aclose()
                    del sessions[sid]
            await asyncio.sleep(600)
    except asyncio.CancelledError:
        pass


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(remove_expired_sessions())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(lifespan=lifespan)


@app.post("/proxy")
async def proxy(request: Request):
    data = await request.json()
    url = data["url"]
    method = data["method"]
    headers = data["headers"]
    body = data["body"]
    proxyurl = data["proxyurl"]
    sessionid = data["sessionid"]
    current_time = time.time()
    async with sessions_lock:
        if sessionid in sessions:
            session, _ = sessions[sessionid]
            sessions[sessionid] = (session, current_time)
        else:
            session = requests.AsyncSession()
            sessionid = str(uuid4())
            sessions[sessionid] = (session, current_time)
    try:
        request_args = {"method": method, "url": url, "headers": headers}
        if body:
            request_args["data"] = body
        if proxyurl:
            request_args["proxies"] = {"http": proxyurl, "https": proxyurl}
        response = await session.request(
            **request_args, impersonate="chrome", timeout=5
        )
    except Exception as e:
        {"status_code": 500, "headers": dict(), "body": ""}
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "content": response.content,
    }