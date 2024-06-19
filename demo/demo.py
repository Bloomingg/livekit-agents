import asyncio
import logging

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatRole,
)
from livekit.agents.translate_assistant import TranslateAssistant
from livekit.plugins import deepgram, openai, silero


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                text=f"""
### Simplified Translation Request

**Task Description:**
As a translator, your task is to translate the provided text into the specified target language. You need to parse the text and target language code from the user's input and then return a result that contains only the translated text.

**Input Example:**
```
text:
Hello, how are you today?
target language:
fr
```

**The output should only include:**
- The translated text.

**Processing Steps:**
1. Parse the user input to obtain the original text and target language code.
2. Use a translation API or simulated translation function to translate the text from the original language to the target language.
3. Output a result that contains only the translated text.

**Output Example:**
```
Bonjour, comment allez-vous aujourd'hui?
```

**Note:**
- Ensure translation accuracy by using reliable translation sources.
- Return an error message for unrecognized language codes.
                """
            )
        ]
    )

    assistant = TranslateAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(
             language="zh-CN"
        ),
        llm=openai.LLM(),
        tts=openai.TTS(voice="alloy"),
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    # await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
