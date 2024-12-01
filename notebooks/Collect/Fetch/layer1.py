# sends requests through curl_cffi

import asyncio
import contextlib
import json
import time

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

async def get_session(sessionid, proxyurl):
    if proxyurl is not None:
        proxies = {"http": proxyurl, "https": proxyurl}
    else:
        proxies = {}
    current_time = time.time()
    async with sessions_lock:
        if sessionid in sessions:
            session, _ = sessions[sessionid]
            sessions[sessionid] = (session, current_time)
            if session.proxies == proxies:
                return session
        session = requests.AsyncSession(proxies=proxies)
        sessions[sessionid] = (session, current_time)
        return session

@app.post("/proxy")
async def proxy(request: Request):
    data = await request.json()
    session = await get_session(data["sessionid"], data.get("proxyurl", None))
    args = {}
    for k in ["method", "url", "headers", "impersonate", "timeout"]:
        if k in data:
            args[k] = data[k]
    if "json" in data:
        args["json"] = json.loads(data["json"])
    try:
        response = await session.request(**args)
    except Exception as e:
        return {"status_code": 500, "headers": dict(), "content": str(args) + "\n" + str(e)}
    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "content": response.content,
    }