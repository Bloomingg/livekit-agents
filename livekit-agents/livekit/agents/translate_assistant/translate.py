from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
import contextvars
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterable, Callable, Literal

from livekit import rtc

from .. import aio, tokenize, transcription, utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad
from . import plotter

logger = logging.getLogger("livekit.agents.translate_assistant")


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


@dataclass(frozen=True)
class _AssistantOptions:
    plotting: bool
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


@dataclass(frozen=True)
class _StartArgs:
    room: rtc.Room
    participant: rtc.RemoteParticipant | str | None


@dataclass
class User:
    track: dict
    language: str

@dataclass
class AudioSource:
    source: dict
    sid: str

@dataclass
class UserMetaData:
    ln: str


_ContextVar = contextvars.ContextVar("voice_assistant_contextvar")


class AssistantContext:
    def __init__(self, assistant: "TranslateAssistant", llm_stream: allm.LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict()
        self._llm_stream = llm_stream

    @staticmethod
    def get_current() -> "AssistantContext":
        return _ContextVar.get()

    @property
    def assistant(self) -> "TranslateAssistant":
        return self._assistant

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> allm.LLMStream:
        return self._llm_stream


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "function_calls_collected",
    "function_calls_finished",
]


class TranslateAssistant(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        vad: avad.VAD,
        stt: astt.STT,
        llm: allm.LLM,
        tts: atts.TTS,
        chat_ctx: allm.ChatContext | None = None,
        fnc_ctx: allm.FunctionContext | None = None,
        interrupt_speech_duration: float = 0.65,
        interrupt_min_words: int = 3,
        base_volume: float = 1.0,
        debug: bool = False,
        plotting: bool = False,
        preemptive_synthesis: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        transcription: bool = True,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        transcription_speed: float = 3.83,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()
        self._opts = _AssistantOptions(
            plotting=plotting,
            debug=debug,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            base_volume=base_volume,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            transcription_speed=transcription_speed,
        )

        # wrap with adapter automatically with default options
        # to override StreamAdapter options, create the adapter manually
        if not tts.streaming_supported:
            tts = atts.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        self._vad, self._tts, self._llm, self._stt = vad, tts, llm, stt
        self._fnc_ctx = fnc_ctx
        self._chat_ctx = chat_ctx or allm.ChatContext()
        self._plotter = plotter.AssistantPlotter(self._loop)

        self._user_map = defaultdict(dict)  # participant identity -> user track & language
        self._audio_source_map = defaultdict(dict)  # language -> audio source & sid

        self._started = False
        self._start_speech_lock = asyncio.Lock()
        self._pending_validation = False

        # tasks
        self._recognize_atask_map: dict[str, asyncio.Task] = dict()
        self._play_atask: asyncio.Task | None = None
        self._tasks = set[asyncio.Task]()

        # playout state
        self._maybe_answer_task: asyncio.Task | None = None
        self._validated_speech: _SpeechData | None = None
        self._answer_speech: _SpeechData | None = None
        self._playout_start_time: float | None = None

        # synthesis state
        self._speech_playing: _SpeechData | None = None  # validated and playing speech
        self._user_speaking, self._agent_speaking = False, False

        self._target_volume = self._opts.base_volume
        self._vol_filter = utils.ExpFilter(0.9, max_val=self._opts.base_volume)
        self._vol_filter.apply(1.0, self._opts.base_volume)
        self._speech_prob = 0.0
        self._transcribed_text, self._interim_text = "", ""
        self._ready_future = asyncio.Future()

    @property
    def chat_context(self) -> allm.ChatContext:
        return self._chat_ctx

    @property
    def started(self) -> bool:
        return self._started

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the translate assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found ivn the room will be selected
        """
        if self.started:
            raise RuntimeError("translate assistant already started")

        self._started = True
        self._start_args = _StartArgs(room=room, participant=participant)

        room.on("track_published", self._on_track_published)
        room.on("track_subscribed", self._on_track_subscribed_sync)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_connected", self._on_participant_connected_sync)
        room.on("participant_metadata_changed", self._on_participant_metadata_changed_sync)

        self._main_atask = asyncio.create_task(self._main_task())

    async def say(
        self,
        source: str | allm.LLMStream | AsyncIterable[str],
    ) -> None:
        """
        Make the assistant say something.
        The source can be a string, an LLMStream or an AsyncIterable[str]

        Args:
            source: the source of the speech
        """
        # await self._wait_ready()

        data = _SpeechData(
            source=source,
            validation_future=asyncio.Future(),
        )
        data.validate_speech()

        print(f"chat with agent text: {source}")

        await self._start_speech(data, interrupt_current_if_possible=False)

        assert self._play_atask is not None
        await self._play_atask

    def on(self, event: EventTypes, callback: Callable | None = None) -> Callable:
        """Register a callback for an event

        Args:
            event: the event to listen to (see EventTypes)
                - user_started_speaking: the user started speaking
                - user_stopped_speaking: the user stopped speaking
                - agent_started_speaking: the agent started speaking
                - agent_stopped_speaking: the agent stopped speaking
                - user_speech_committed: the user speech was committed to the chat context
                - agent_speech_committed: the agent speech was committed to the chat context
                - agent_speech_interrupted: the agent speech was interrupted
                - function_calls_collected: received the complete set of functions to be executed
                - function_calls_finished: all function calls have been completed
            callback: the callback to call when the event is emitted
        """
        return super().on(event, callback)

    async def aclose(self, wait: bool = True) -> None:
        """
        Close the translate assistant

        Args:
            wait: whether to wait for the current speech to finish before closing
        """
        if not self.started:
            return

        self._ready_future.cancel()

        self._start_args.room.off("track_published", self._on_track_published)
        self._start_args.room.off("track_subscribed", self._on_track_subscribed_sync)
        self._start_args.room.off("track_unsubscribed", self._on_track_unsubscribed)
        self._start_args.room.off(
            "participant_connected", self._on_participant_connected_sync
        )
        self._start_args.room.off("participant_metadata_changed", self._on_participant_metadata_changed_sync)

        self._plotter.terminate()

        with contextlib.suppress(asyncio.CancelledError):
            self._main_atask.cancel()
            await self._main_atask

        if self._recognize_atask_map is not None:
            for _, task in self._recognize_atask_map.items():
                task.cancel()

        if not wait:
            if self._play_atask is not None:
                self._play_atask.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            if self._play_atask is not None:
                await self._play_atask

            if self._recognize_atask_map is not None:
                for _, task in self._recognize_atask_map.items():
                    task.cancel()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """
        Main task is publising the agent audio track and run the update loop
        """
        if self._opts.plotting:
            self._plotter.start()
    
        if self._start_args.participant is not None:
            if isinstance(self._start_args.participant, rtc.RemoteParticipant):
                await self._link_participant(self._start_args.participant.identity)
            else:
                await self._link_participant(self._start_args.participant)
        else:
            # no participant provided, try to find the first in the room
            print(self._start_args.room.participants.values())
            for participant in self._start_args.room.participants.values():
                await self._link_participant(participant.identity)
                break

        self._ready_future.set_result(None)

        # Loop running each 10ms to do the following:
        #  - Update the volume based on the user speech probability
        #  - Decide when to interrupt the agent speech
        #  - Decide when to validate the user speech (starting the agent answer)
        speech_prob_avg = utils.MovingAverage(100)
        speaking_avg_validation = utils.MovingAverage(150)
        interruption_speaking_avg = utils.MovingAverage(
            int(self._opts.int_speech_duration * 100)
        )

        interval_10ms = aio.interval(0.01)

        vad_pw = 2.4  # TODO(theomonnom): should this be exposed?
        while True:
            await interval_10ms.tick()

            speech_prob_avg.add_sample(self._speech_prob)
            speaking_avg_validation.add_sample(int(self._user_speaking))
            interruption_speaking_avg.add_sample(int(self._user_speaking))

            bvol = self._opts.base_volume
            self._target_volume = max(0, 1 - speech_prob_avg.get_avg() * vad_pw) * bvol

            if self._pending_validation:
                if speaking_avg_validation.get_avg() <= 0.05:
                    self._validate_answer_if_needed()

            if self._opts.plotting:
                self._plotter.plot_value("raw_t_vol", self._target_volume)
                self._plotter.plot_value("vol", self._vol_filter.filtered())

    def _parse_metadata(self, metadata: str) -> UserMetaData:
        """
        Parse metadata from the user input
        """
        try:
            data_dict = json.loads(metadata)
            ln_value = data_dict.get("language", "disabled")
            return UserMetaData(ln=ln_value)
        except Exception:
            return UserMetaData(ln="disabled")
        
    async def _prepare_user_map(self, participant: Any, track: Any = None, old_ln: str = None) -> str:
        """
        Prepare the user map with the participant and track
        """
        ln = self._parse_metadata(participant.metadata).ln

        if ln != "disabled":
            self._user_map[participant.identity] = {
                "track": track,
                "language": ln
            }
        elif participant.identity in self._user_map:
                task = self._recognize_atask_map.get(participant.identity)
                if task is not None:
                    task.cancel()
                    del self._recognize_atask_map[participant.identity]
                del self._user_map[participant.identity]
        if old_ln is not None and ln != old_ln and old_ln in self._audio_source_map:
            await self._start_args.room.local_participant.unpublish_track(self._audio_source_map[old_ln]['sid'])
            del self._audio_source_map[old_ln]
        return ln

    async def _link_participant(self, identity: str) -> None:
        p = self._start_args.room.participants_by_identity.get(identity)
        assert p is not None, "_link_participant should be called with a valid identity"
        
        ln = await self._prepare_user_map(p)
        print(f"_link_participant ln-{ln} identity-{identity} metadata-{p.metadata}")
        if ln == "disabled":
            return

        if (ln not in self ._audio_source_map):
            self._audio_source_map[ln]['source'] = rtc.AudioSource(
                self._tts.sample_rate, self._tts.num_channels
            )
            track = rtc.LocalAudioTrack.create_audio_track(
                f"translate_voice_{ln}", self._audio_source_map[ln]['source']
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            pub = await self._start_args.room.local_participant.publish_track(
                track, options
            )
            self._audio_source_map[ln]['sid'] = pub.sid
            print(f"linking participant {identity} ln {ln} sid {self._audio_source_map[ln]['sid']}")

        self._log_debug(f"linking participant {identity}")

        for pub in p.tracks.values():
            if pub.subscribed:
                self._on_track_subscribed_sync(pub.track, pub, p)  # type: ignore
            else:
                self._on_track_published(pub, p)

    def _on_participant_connected_sync(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._on_participant_connected(participant))

    def _on_participant_metadata_changed_sync(self, participant: rtc.RemoteParticipant, old_metadata: str, new_metadata: str):
        print(f"_on_participant_metadata_changed old_metadata {old_metadata} new_metadata {new_metadata}")
        asyncio.create_task(self._on_participant_metadata_changed(participant, old_metadata))

    async def _on_participant_metadata_changed(self, participant: rtc.RemoteParticipant, old_metadata: str):
        # if participant.identity in self._user_map:
            old_ln = self._parse_metadata(old_metadata).ln
            ln = await self._prepare_user_map(participant=participant,old_ln=old_ln)
            print(f"old_ln {old_ln} ln {ln}")
            if ln != "disabled" and old_ln != ln:
                await self._link_participant(participant.identity)

    async def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        print(f"_on_participant_connected {participant.identity} {participant.identity not in self._user_map}")
        if participant.identity not in self._user_map:
            await self._link_participant(participant.identity)

    def _on_track_published(
        self, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        if (
            pub.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        if not pub.subscribed:
            pub.set_subscribed(True)

    def _on_track_subscribed_sync(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(self._on_track_subscribed(track,pub,participant))

    async def _on_track_subscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            (participant.identity in self._user_map and self._user_map[participant.identity]['track'] is not None)
            or pub.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        ln = await self._prepare_user_map(participant, track)
        if ln == "disabled":
            return
        
        self._log_debug("starting listening to user microphone")
        print(f"starting listening to user microphone {participant.identity}")
        task = asyncio.create_task(
            self._recognize_task(rtc.AudioStream(track), participant.identity)
        )
        self._recognize_atask_map[participant.identity] = task

    def _on_track_unsubscribed(
        self,
        track: rtc.RemoteTrack,
        pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            participant.identity not in self._user_map
            or pub.source != rtc.TrackSource.SOURCE_MICROPHONE
        ):
            return

        # user microphone unsubscribed, (participant disconnected/track unpublished)
        self._log_debug("user microphone not available anymore")
        print(f"user microphone not available anymore {participant.identity}")
        assert (
            self._recognize_atask_map is not None
        ), "recognize task should be running when user_track was set"
        task = self._recognize_atask_map.get(participant.identity)
        if task is not None:
            task.cancel()
            del self._recognize_atask_map[participant.identity]
        del self._user_map[participant.identity]

    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream, identity: str) -> None:
        """
        Receive the frames from the user audio stream and do the following:
         - do Translate Activity Detection (VAD)
         - do Speech-to-Text (STT)
        """
        # assert (
        #     self._user_identity is not None
        # ), "user identity should be set before recognizing"
        print(f"_recognize_task {self._user_map[identity]['language']}")
        vad_stream = self._vad.stream()
        stt_stream = self._stt.stream(language=self._user_map[identity]['language'])

        stt_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            stt_forwarder = transcription.STTSegmentsForwarder(
                room=self._start_args.room,
                participant=identity,
                track=self._user_map[identity]['track'],
            )

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                 if len(self._user_map) == 2:
                    other_participant = next((k for k in self._user_map if k != identity), None)
                    if other_participant and self._user_map[other_participant]['language'] != self._user_map[identity]['language']:
                        stt_stream.push_frame(ev.frame)
                        vad_stream.push_frame(ev.frame)

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == avad.VADEventType.START_OF_SPEECH:
                    self._log_debug("user started speaking")
                    self._plotter.plot_event("user_started_speaking")
                    self._user_speaking = True
                    self.emit("user_started_speaking")
                elif ev.type == avad.VADEventType.INFERENCE_DONE:
                    self._plotter.plot_value("vad_raw", ev.raw_inference_prob)
                    self._plotter.plot_value("vad_smoothed", ev.probability)
                    self._plotter.plot_value("vad_dur", ev.inference_duration * 1000)
                    self._speech_prob = ev.raw_inference_prob
                elif ev.type == avad.VADEventType.END_OF_SPEECH:
                    self._log_debug(f"user stopped speaking {ev.duration:.2f}s")
                    self._plotter.plot_event("user_started_speaking")
                    self._pending_validation = True
                    self._user_speaking = False
                    self.emit("user_stopped_speaking")

        async def _stt_stream_co() -> None:
            async for ev in stt_stream:
                stt_forwarder.update(ev)
                if ev.type == astt.SpeechEventType.FINAL_TRANSCRIPT:
                    targetln = ''
                    for _, value in self._user_map.items():
                        if value['language'] != self._user_map[identity]['language']:
                            targetln = value['language']
                            break
                    print(f"stt_stream_co {ev.alternatives[0].text} targetln {targetln}")
                    if targetln:
                        self._on_final_transcript(ev.alternatives[0].text, targetln)
                elif ev.type == astt.SpeechEventType.INTERIM_TRANSCRIPT:
                    # interim transcript is used in combination with VAD
                    # to interrupt the current speech.
                    # (can be disabled by setting int_min_words to 0)
                    self._interim_text = ev.alternatives[0].text
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

    def _on_final_transcript(self, text: str, language: str) -> None:
        self._transcribed_text += text
        self._log_debug(f"received final transcript: {self._transcribed_text}")

        # to create an llm stream we need an async context
        # setting it to "" and will be updated inside the _answer_task below
        # (this function can't be async because we don't want to block _update_co)
        self._answer_speech = _SpeechData(
            source="",
            language=language,
            validation_future=asyncio.Future(),
            original_text=self._transcribed_text,
        )

        text = f"""
text:
{self._transcribed_text}
target language:
{language}
"""
        print(f"chat with user text: {text}")
        # this speech may not be validated, so we create a copy
        # of our context to add the new user message
        copied_ctx = self._chat_ctx.copy()
        copied_ctx.messages.append(
            allm.ChatMessage(
                text=text,
                role=allm.ChatRole.USER,
            )
        )

        if self._maybe_answer_task is not None:
            self._maybe_answer_task.cancel()

        async def _answer_task(ctx: allm.ChatContext, data: _SpeechData) -> None:
            try:
                data.source = await self._llm.chat(ctx, fnc_ctx=self._fnc_ctx)
                await self._start_speech(data, interrupt_current_if_possible=False)
            except Exception:
                logger.exception("error while answering")

        t = asyncio.create_task(_answer_task(copied_ctx, self._answer_speech))
        self._maybe_answer_task = t
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _validate_answer_if_needed(self) -> None:
        """
        Check whether the current pending answer to the user should be validated (played)
        """
        if self._answer_speech is None:
            return

        self._pending_validation = False
        self._transcribed_text = self._interim_text = ""
        self._answer_speech.validate_speech()
        self._log_debug("user speech validated")

    async def _start_speech(
        self, data: _SpeechData, *, interrupt_current_if_possible: bool
    ) -> None:
        # await self._wait_ready()

        async with self._start_speech_lock:
            # interrupt the current speech if possible, otherwise wait before playing the new speech
            if self._play_atask is not None:
                if self._validated_speech is not None:
                    if (
                        interrupt_current_if_possible
                    ):
                        logger.debug("_start_speech - interrupting current speech")

                else:
                    # pending speech isn't validated yet, OK to cancel
                    self._play_atask.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await self._play_atask

            self._play_atask = asyncio.create_task(
                self._play_speech_if_validated_task(data)
            )

    @utils.log_exceptions(logger=logger)
    async def _play_speech_if_validated_task(self, data: _SpeechData) -> None:
        """
        Start synthesis and playout the speech only if validated
        """
        self._log_debug(f"play_speech_if_validated {data.original_text}")
        print(f"play_speech_if_validated {data.original_text} sid {self._audio_source_map[data.language]['sid']}")

        # reset volume before starting a new speech
        self._vol_filter.reset()
        playout_tx, playout_rx = aio.channel()  # playout channel

        tts_forwarder = utils._noop.Nop()
        if self._opts.transcription:
            tts_forwarder = transcription.TTSSegmentsForwarder(
                room=self._start_args.room,
                participant=self._start_args.room.local_participant,
                track=self._audio_source_map[data.language]['sid'],
                sentence_tokenizer=self._opts.sentence_tokenizer,
                word_tokenizer=self._opts.word_tokenizer,
                hyphenate_word=self._opts.hyphenate_word,
                speed=self._opts.transcription_speed,
            )

        if not self._opts.preemptive_synthesis:
            await data.validation_future

        tts_co = self._synthesize_task(data, playout_tx, tts_forwarder)
        _synthesize_task = asyncio.create_task(tts_co)
        print('_synthesize_task')
        try:
            # wait for speech validation before playout
            await data.validation_future

            # validated!
            self._validated_speech = data
            self._playout_start_time = time.time()

            self._log_debug("starting playout")
            print("starting playout")
            await self._playout_co(playout_rx, tts_forwarder, data)
            
            self._log_debug("playout finished")
        finally:
            self._validated_speech = None
            with contextlib.suppress(asyncio.CancelledError):
                _synthesize_task.cancel()
                await _synthesize_task

            # make sure that _synthesize_task is finished before closing the transcription
            # forwarder. pushing text/audio to the forwarder after closing it will raise an exception
            await tts_forwarder.aclose()
            self._log_debug("play_speech_if_validated_task finished")

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

        start_time = time.time()
        first_frame = True
        audio_duration = 0.0
        try:
            async for audio in self._tts.synthesize(text):
                if first_frame:
                    first_frame = False
                    dt = time.time() - start_time
                    self._log_debug(f"tts first frame in {dt:.2f}s")
                    print(f"tts first frame in {dt:.2f}s")

                frame = audio.data
                audio_duration += frame.samples_per_channel / frame.sample_rate
                
                playout_tx.send_nowait(frame)
                tts_forwarder.push_audio(frame)

        finally:
            tts_forwarder.mark_audio_segment_end()
            playout_tx.close()
            self._log_debug(f"tts finished synthesising {audio_duration:.2f}s of audio")
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
            chat = rtc.ChatManager(self._start_args.room)
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

    @utils.log_exceptions(logger=logger)
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
            assistant_ctx = AssistantContext(self, llm_stream)
            token = _ContextVar.set(assistant_ctx)

            async def _forward_llm_chunks():
                async for chunk in llm_stream:
                    alt = chunk.choices[0].delta.content
                    if not alt:
                        continue
                    yield alt

            await self._synthesize_streamed_speech_co(
                data, playout_tx, _forward_llm_chunks(), tts_forwarder
            )

            if len(llm_stream.called_functions) > 0:
                self.emit("function_calls_collected", assistant_ctx)

            await llm_stream.aclose()

            if len(llm_stream.called_functions) > 0:
                self.emit("function_calls_finished", assistant_ctx)

            _ContextVar.reset(token)
        else:
            await self._synthesize_streamed_speech_co(
                data, playout_tx, data.source, tts_forwarder
            )

    async def _playout_co(
        self,
        playout_rx: aio.ChanReceiver[rtc.AudioFrame],
        tts_forwarder: transcription.TTSSegmentsForwarder | utils._noop.Nop,
        speech_data: _SpeechData,
    ) -> None:
        """
        Playout audio with the current volume.
        The playout_rx is streaming the synthesized speech from the TTS provider to minimize latency
        """
        assert (
            speech_data.language in self._audio_source_map
        ), "audio source should be set before playout"

        first_frame = True
        async for frame in playout_rx:
            if first_frame:
                self._log_debug("agent started speaking")
                print("agent started speaking")
                self._plotter.plot_event("agent_started_speaking")
                self._agent_speaking = True
                self.emit("agent_started_speaking")
                tts_forwarder.segment_playout_started()  # we have only one segment
                first_frame = False

            await self._audio_source_map[speech_data.language]['source'].capture_frame(
               frame
            )
               

        if not first_frame:
            self._log_debug("agent stopped speaking")
            print(f"agent stopped speaking")
            tts_forwarder.segment_playout_finished()

            self._plotter.plot_event("agent_stopped_speaking")
            self._agent_speaking = False
            self.emit("agent_stopped_speaking")

    def _log_debug(self, msg: str, **kwargs) -> None:
        if self._opts.debug:
            logger.debug(msg, **kwargs)

    async def _wait_ready(self) -> None:
        """Wait for the assistant to be fully started"""
        await self._ready_future
