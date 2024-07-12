import asyncio
import contextlib
from dataclasses import dataclass
import json
from typing import AsyncIterable, Any, Callable
from .. import aio, transcription, tokenize, utils

from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad

from livekit import rtc

@dataclass(frozen=True)
class AssistantOptions:
    debug: bool
    int_speech_duration: float
    int_min_words: int
    base_volume: float
    transcription: bool
    preemptive_synthesis: bool
    word_tokenizer: tokenize.WordTokenizer
    sentence_tokenizer: tokenize.SentenceTokenizer
    hyphenate_word: Callable[[str], list[str]]
    transcription_speed: float

@dataclass
class _SpeechData:
    source: str | allm.LLMStream | AsyncIterable[str]
    validation_future: asyncio.Future[None]  # validate the speech for playout
    validated: bool = False
    original_text: str | None = None
    translated_text: str = ""
    language: str = "en-US"

    def validate_speech(self) -> None:
        self.validated = True
        with contextlib.suppress(asyncio.InvalidStateError):
            self.validation_future.set_result(None)

@dataclass
class AudioSourceData:
    source: rtc.AudioSource
    sid: str

class SpeechHandler:
    def __init__(
        self, 
        language: str, 
        room: rtc.Room, 
        llm: allm.LLM, 
        vad: avad.VAD,
        stt: astt.STT,
        tts: atts.TTS,
        chat_ctx: allm.ChatContext, 
        opts: AssistantOptions, 
        track: rtc.RemoteTrack,
        get_target_language_fnc: Callable[[str], str | None],
        get_target_audio_source_fnc: Callable[[str], AudioSourceData | None]
    ):
        self._transcribed_text = ""
        self._pending_validation = False
        self._user_speaking = False
        self._answer_speech: _SpeechData | None = None
        self._play_atask: asyncio.Task | None = None
        self._start_speech_lock = asyncio.Lock()
        self._tasks = set[asyncio.Task]()
        self._loop_check_task: asyncio.Task | None = None
        self._target_language: str | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._sid: str | None = None
        self._identity: str | None = None

        self._language = language
        self._room = room
        self._llm = llm
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._chat_ctx = chat_ctx
        self._opts = opts

        self._track = track

        self._get_target_language_fnc = get_target_language_fnc
        self._get_target_audio_source_fnc = get_target_audio_source_fnc

    async def recognize_task(self, audio_stream: rtc.AudioStream, identity: str) -> None:
        """
        Receive the frames from the user audio stream and do the following:
         - do Translate Activity Detection (VAD)
         - do Speech-to-Text (STT)
        """
        print(f"recognize_task {identity}")
        if self._loop_check_task is None:
            self._loop_check_task = asyncio.create_task(self._loop_check())

        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream(language=self._language)
        self._identity = identity

        stt_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            stt_forwarder = transcription.STTSegmentsForwarder(
                room=self._room,
                participant=identity,
                track=self._track
            )

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                if self._target_language is None:
                    target_language = self._get_target_language_fnc(self._identity)
                    self._target_language = target_language
                if self._target_language:
                    stt_stream.push_frame(ev.frame)
                    vad_stream.push_frame(ev.frame)

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == avad.VADEventType.START_OF_SPEECH:
                    self._user_speaking = True
                elif ev.type == avad.VADEventType.END_OF_SPEECH:
                    self._pending_validation = True
                    self._user_speaking = False

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)
                if ev.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                    print(f"_target_language {self._target_language}  text {ev.alternatives[0].text}")
                    if self._target_language:
                        self._on_final_transcript(ev.alternatives[0].text)
                elif ev.type == astt.SpeechEventType.END_OF_SPEECH:
                    self._pending_validation = True
        try:
            await asyncio.gather(
                _audio_stream_co(),
                _vad_stream_co(),
                _stt_stream_co(),
            )
        finally:
            await asyncio.gather(
                stt_forwarder.aclose(wait=False),
                stt_stream.aclose(wait=False),
                vad_stream.aclose(wait=False),
            )

    async def _loop_check(self) -> None:
        # Loop running each 10ms to do the following:
        #  - Decide when to validate the user speech (starting the agent answer)
        speaking_avg_validation = utils.MovingAverage(150)

        interval_10ms = aio.interval(0.01)
        while True:
            await interval_10ms.tick()

            speaking_avg_validation.add_sample(int(self._user_speaking))

            if self._pending_validation:
                # The larger the value, the earlier the stop is triggered.
                if speaking_avg_validation.get_avg() <= 0.03:
                    self._validate_answer_if_needed()

    def _validate_answer_if_needed(self):
        if self._answer_speech is None or self._transcribed_text == "":
            return

        self._pending_validation = False
        print(f"user speech validated {self._transcribed_text}")
        self._transcribed_text = ""
        self._answer_speech.validate_speech()
        self._translate_and_play()

    def _on_final_transcript(self, text: str) -> None:
        self._transcribed_text += text
        
        # to create an llm stream we need an async context
        # setting it to "" and will be updated inside the _answer_task below
        # (this function can't be async because we don't want to block _update_co)
        self._answer_speech = _SpeechData(
            source="",
            language=self._target_language,
            validation_future=asyncio.Future(),
            original_text=self._transcribed_text,
        )

    def _translate_and_play(self):
        text = f"text:\n{self._answer_speech.original_text}\ntarget language:\n{self._answer_speech.language}"
        copied_ctx = self._chat_ctx.copy()
        copied_ctx.messages.append(allm.ChatMessage(text=text, role=allm.ChatRole.USER))

        async def _answer_task(ctx: allm.ChatContext, data: _SpeechData):
            # try:
                data.source = await self._llm.chat(ctx, fnc_ctx=None)
                await self._start_speech(data)
            # except Exception as e:
            #     print("Error while answering")

        t = asyncio.create_task(_answer_task(copied_ctx, self._answer_speech))
        self._tasks.add(t)
        t.add_done_callback(lambda x: self._tasks.discard(x))

    async def _start_speech(self, data: _SpeechData):
        async with self._start_speech_lock:
            if self._play_atask is not None:
                print("wait _play_atask")
                await self._play_atask

            self._play_atask = asyncio.create_task(self._play_speech_if_validated_task(data))
    
    async def _play_speech_if_validated_task(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        print(f"play_speech_if_validated {data.original_text}")

        if self._audio_source is None:
            audio_source_data = self._get_target_audio_source_fnc(self._identity)
            if audio_source_data is None:
                return
            self._audio_source = audio_source_data.source
            self._sid = audio_source_data.sid

        playout_tx, playout_rx = aio.channel()  # playout channel

        tts_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            tts_forwarder = transcription.TTSSegmentsForwarder(
                room=self._room,
                participant=self._room.local_participant,
                track=self._sid,
                sentence_tokenizer=self._opts.sentence_tokenizer,
                word_tokenizer=self._opts.word_tokenizer,
                hyphenate_word=self._opts.hyphenate_word,
                speed=self._opts.transcription_speed,
            )

        if not self._opts.preemptive_synthesis:
            await data.validation_future

        tts_co = self._synthesize_task(data, playout_tx, tts_forwarder)
        _synthesize_task = asyncio.create_task(tts_co)
        try:
            # wait for speech validation before playout
            await data.validation_future

            await self._playout_co(playout_rx, tts_forwarder)
            print("starting playout")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                _synthesize_task.cancel()
                await _synthesize_task

            # make sure that _synthesize_task is finished before closing the transcription
            # forwarder. pushing text/audio to the forwarder after closing it will raise an exception
            await tts_forwarder.aclose()

    async def _synthesize_speech_co(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        text: str,
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """synthesize speech from a string"""
        data.translated_text += text
        tts_forwarder.push_text(text)
        tts_forwarder.mark_text_segment_end()

        first_frame = True
        audio_duration = 0.0
        try:
            async for audio in self._tts.synthesize(text):
                if first_frame:
                    first_frame = False

                frame = audio.data
                audio_duration += frame.samples_per_channel / frame.sample_rate
                
                playout_tx.send_nowait(frame)
                tts_forwarder.push_audio(frame)

        finally:
            tts_forwarder.mark_audio_segment_end()
            playout_tx.close()
            print(f"tts finished synthesising {audio_duration:.2f}s of audio text: {text}")

    async def _synthesize_streamed_speech_co(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        streamed_text: AsyncIterable[str],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """synthesize speech from streamed text"""

        async def _read_generated_audio_task():
            first_frame = True
            audio_duration = 0.0
            async for event in tts_stream:
                if event.type == atts.SynthesisEventType.AUDIO:
                    if first_frame:
                        first_frame = False

                    assert event.audio is not None
                    frame = event.audio.data
                    audio_duration += frame.samples_per_channel / frame.sample_rate
                    tts_forwarder.push_audio(frame)
                    playout_tx.send_nowait(frame)


        # otherwise, stream the text to the TTS
        tts_stream = self._tts.stream()
        read_task = asyncio.create_task(_read_generated_audio_task())

        try:
            async for seg in streamed_text:
                data.translated_text += seg
                tts_forwarder.push_text(seg)
                tts_stream.push_text(seg)

        finally:
            print(f"llm return text {data.translated_text}")
            chat = rtc.ChatManager(self._room)
            message = {
                "language": data.language,
                "text": data.translated_text
            }
            message_str = json.dumps(message)
            await chat.send_message(message_str)
            tts_forwarder.mark_text_segment_end()
            tts_stream.mark_segment_end()

            await tts_stream.aclose()
            await read_task
            
            tts_forwarder.mark_audio_segment_end()
            playout_tx.close()

    async def _synthesize_task(
        self,
        data: _SpeechData,
        playout_tx: aio.ChanSender[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """Synthesize speech from the source. Also run LLM inference when needed"""
        if isinstance(data.source, str):
            await self._synthesize_speech_co(
                data, playout_tx, data.source, tts_forwarder
            )
        elif isinstance(data.source, allm.LLMStream):
            llm_stream = data.source

            async def _forward_llm_chunks():
                async for chunk in llm_stream:
                    alt = chunk.choices[0].delta.content
                    if not alt:
                        continue
                    yield alt

            await self._synthesize_streamed_speech_co(
                data, playout_tx, _forward_llm_chunks(), tts_forwarder
            )

            await llm_stream.aclose()
        else:
            await self._synthesize_streamed_speech_co(
                data, playout_tx, data.source, tts_forwarder
            )

    async def _playout_co(
        self,
        playout_rx: aio.ChanReceiver[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
    ) -> None:
        """
        Playout audio with the current volume.
        The playout_rx is streaming the synthesized speech from the TTS provider to minimize latency
        """
        first_frame = True
        async for frame in playout_rx:
            if first_frame:
                print("agent started speaking")
                tts_forwarder.segment_playout_started()  # we have only one segment
                first_frame = False

            await self._audio_source.capture_frame(
               frame
            )
               

        if not first_frame:
            print(f"agent stopped speaking")
            tts_forwarder.segment_playout_finished()
