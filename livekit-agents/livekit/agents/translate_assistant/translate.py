from __future__ import annotations

import asyncio
from collections import defaultdict
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Callable

from livekit import rtc

from .. import tokenize, utils
from .. import llm as allm
from .. import stt as astt
from .. import tts as atts
from .. import vad as avad

from .speech_handler import SpeechHandler, AssistantOptions, AudioSourceData

logger = logging.getLogger("livekit.agents.translate_assistant")

@dataclass(frozen=True)
class _StartArgs:
    room: rtc.Room
    participant: rtc.RemoteParticipant | str | None

@dataclass
class UserMetaData:
    language: str

@dataclass
class UserData:
    hander: SpeechHandler
    language: str
    audio_source: rtc.AudioSource
    sid: str

class TranslateAssistant():
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
        preemptive_synthesis: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        transcription: bool = True,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(),
        hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word,
        transcription_speed: float = 3.83,
    ) -> None:
        super().__init__()
        self._opts = AssistantOptions(
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

        self._user_map: dict[str, UserData] = defaultdict(dict)  # participant identity -> user data

        self._started = False

        # tasks
        self._recognize_atask_map: dict[str, asyncio.Task] = dict()

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
        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)
        room.on("participant_connected", self._on_participant_connected_sync)

        self._main_atask = asyncio.create_task(self._main_task())

    async def aclose(self, wait: bool = True) -> None:
        """
        Close the translate assistant

        Args:
            wait: whether to wait for the current speech to finish before closing
        """
        if not self.started:
            return
        
        self._start_args.room.off("track_published", self._on_track_published)
        self._start_args.room.off("track_subscribed", self._on_track_subscribed)
        self._start_args.room.off("track_unsubscribed", self._on_track_unsubscribed)
        self._start_args.room.off(
            "participant_connected", self._on_participant_connected_sync
        )

        with contextlib.suppress(asyncio.CancelledError):
            self._main_atask.cancel()
            await self._main_atask

        if self._recognize_atask_map is not None:
            for _, task in self._recognize_atask_map.items():
                task.cancel()


        with contextlib.suppress(asyncio.CancelledError):
            if self._recognize_atask_map is not None:
                for _, task in self._recognize_atask_map.items():
                    task.cancel()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        """
        Main task is publising the agent audio track and run the update loop
        """
        if self._start_args.participant is not None:
            if isinstance(self._start_args.participant, rtc.RemoteParticipant):
                await self._link_participant(self._start_args.participant.identity)
            else:
                await self._link_participant(self._start_args.participant)
        else:
            print(f"parts {len(self._start_args.room.participants.values())}")
            # no participant provided, try to find the first in the room
            for participant in self._start_args.room.participants.values():
                await self._link_participant(participant.identity)

    def _parse_metadata(self, metadata: str) -> UserMetaData:
        """
        Parse metadata from the user input
        """
        try:
            data_dict = json.loads(metadata)
            language_value = data_dict.get("language", "disabled")
            return UserMetaData(language=language_value)
        except Exception:
            return UserMetaData(language="disabled")

    def _get_target_language(self, identity: str) -> str | None:
        """
        Get the target language excluded the identity
        """
        other_identity = next((k for k in self._user_map if k != identity), None)
        if other_identity is not None:
            return self._user_map[other_identity].language
        return None
    
    def _get_target_audio_source(self, identity: str) -> AudioSourceData | None:
        """
        Get the target audio source excluded the identity
        """
        other_identity = next((k for k in self._user_map if k != identity), None)
        if other_identity is not None:
            return AudioSourceData(
                sid=self._user_map[other_identity].sid,
                source=self._user_map[other_identity].audio_source
            )
        return None

    async def _link_participant(self, identity: str) -> None:
        p = self._start_args.room.participants_by_identity.get(identity)
        assert p is not None, "_link_participant should be called with a valid identity"
        
        language = self._parse_metadata(p.metadata).language
        print(f"_link_participant language-{language} identity-{identity} metadata-{p.metadata}")
        if language == "disabled":
            return

        if identity not in self._user_map:
            audio_source = rtc.AudioSource(
                self._tts.sample_rate, self._tts.num_channels
            )
            track = rtc.LocalAudioTrack.create_audio_track(
                f"translate_voice_{language}", audio_source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            pub = await self._start_args.room.local_participant.publish_track(
                track, options
            )
            self._user_map[identity] = UserData(
                sid=pub.sid,
                language=language,
                audio_source=audio_source,
                hander=SpeechHandler(
                    language=language,
                    llm=self._llm,
                    vad=self._vad,
                    stt=self._stt,
                    tts=self._tts,
                    opts=self._opts,
                    chat_ctx=self._chat_ctx,
                    room=self._start_args.room,
                    track=track,
                    get_target_language_fnc=self._get_target_language,
                    get_target_audio_source_fnc=self._get_target_audio_source
                )
            )

        for pub in p.tracks.values():
            if pub.subscribed:
                self._on_track_subscribed(pub.track, pub, p)  # type: ignore
            else:
                self._on_track_published(pub, p)

    def _on_participant_connected_sync(self, participant: rtc.RemoteParticipant):
        asyncio.create_task(self._on_participant_connected(participant))

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

    def _on_track_subscribed(
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

        language = self._parse_metadata(participant.metadata).language
        if language == "disabled":
            return
        
        print(f"starting listening to user microphone {participant.identity}")
        task = asyncio.create_task(
            self._user_map[participant.identity].hander.recognize_task(rtc.AudioStream(track), participant.identity)
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
        print(f"user microphone not available anymore {participant.identity}")
        assert (
            self._recognize_atask_map is not None
        ), "recognize task should be running when user_track was set"
        task = self._recognize_atask_map.get(participant.identity)
        if task is not None:
            task.cancel()
            del self._recognize_atask_map[participant.identity]
        del self._user_map[participant.identity]