from .dummy import DummyASR

__all__ = ["DummyASR"]

# FasterWhisperASR is intentionally NOT imported here: importing it pulls in
# faster-whisper/CTranslate2. The factory imports it lazily only when selected,
# so the dummy path stays dependency-free. Import directly if you need it:
#   from s2st.asr.faster_whisper import FasterWhisperASR
