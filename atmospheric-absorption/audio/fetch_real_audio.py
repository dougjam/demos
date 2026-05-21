"""
Fetch real public-domain recordings for the music and voice presets, then
convert to mono 48 kHz WAV trimmed and peak-normalized to -6 dBFS.

Sources (Wikimedia Commons, U.S. public domain — both recordings predate
1926; per current U.S. copyright law sound recordings published before
1926 are in the public domain):

  music.wav  <-  Original Dixieland Jass Band, "Tiger Rag" (1918)
                 https://commons.wikimedia.org/wiki/File:Tiger_Rag_ODJB.ogg
                 Full jazz band: cornet, clarinet, trombone, piano, drums.

  voice.wav  <-  Neil Armstrong, "That's one small step..." (Apollo 11,
                 1969). Spoken from the lunar surface, transmitted over
                 NASA S-band radio; carries the characteristic narrow-
                 band, heavily compressed PA-loudspeaker quality of
                 quarter-million-mile space comms.
                 https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg

  thunder.wav <- "Very close thunder cracks" by Delorean76. Real close-
                 strike thunder; sharp transient followed by natural
                 rolling decay.
                 CC BY-SA 3.0. Attribution noted in ATTRIBUTION.md.
                 https://commons.wikimedia.org/wiki/File:Very_close_thunder_cracks.flac

  gunshot.wav <- "Gunshot Sounds" pack by Tabasco (OpenGameArt.org),
                 CC0. Real outdoor target-range recordings of a CZ-52
                 pistol, Mosin Nagant rifle, SKS, and shotgun. We use
                 the first shot from mosin.wav (a Mosin Nagant rifle,
                 ~7.62 mm, loud crack-and-decay transient).
                 https://opengameart.org/content/gunshot-sounds

  fan.wav     <- "data center ambient noise" by Andron827 — real
                 close-mic'd recording in a computer test lab.
                 Constant broadband drone, no transients. CC0 (no
                 attribution required, but credited in
                 ATTRIBUTION.md). The original 2:02 WAV requires a
                 Freesound login to download; we fetch the equivalent
                 publicly hosted HQ MP3 preview (320 kbps) and
                 transcode/trim/normalize from there. The accuracy
                 cost is negligible at our sample rate and the file
                 is CC0 so format doesn't matter.
                 https://freesound.org/people/Andron827/sounds/646877/

  frogs.wav    <- "Frogs Toads and Night Birds in Utah" by Danjocross.
                  Nighttime chorus in a small canyon pothole near Hite,
                  Utah; calm conditions, no wind/traffic. CC0. Fetched
                  via the public HQ MP3 preview (same reason as fan).
                  https://freesound.org/people/Danjocross/sounds/503211/

  sealions.wav <- "Colony of sea lions on land barking and vocalizing"
                  by RavenWolfProds. ~200 sea lions in Crescent City,
                  California — barks, growls, snores, huffs. CC0.
                  Fetched via HQ MP3 preview.
                  https://freesound.org/people/RavenWolfProds/sounds/503679/

Re-run if any source URL changes.
"""
from __future__ import annotations
import re
import subprocess
import urllib.request
import ssl
import zipfile
from pathlib import Path

HERE = Path(__file__).parent

SOURCES = {
    "music": {
        "url":  "https://upload.wikimedia.org/wikipedia/commons/e/ea/Tiger_Rag_ODJB.ogg",
        "raw":  "music_src.ogg",
        "wav":  "music.wav",
        "start": 30,     # seconds into the file (skip intro)
        "dur":   7.0,    # clip length (seconds)
    },
    "voice": {
        "url":  "https://upload.wikimedia.org/wikipedia/commons/d/dd/Armstrong_Small_Step.ogg",
        "raw":  "voice_src.ogg",
        "wav":  "voice.wav",
        # The OGG opens with ~14 s of preamble ("I'm going to step off
        # the LEM now..."). The famous "That's one small step for a
        # man, one giant leap for mankind" line sits at 14.8-23.8 s.
        # 9 s is over the spec's 3-8 s guideline by 1 s, but anything
        # shorter clips off "...mankind".
        "start": 14.8,
        "dur":   9.0,
    },
    "thunder": {
        "url":  "https://upload.wikimedia.org/wikipedia/commons/f/f5/Very_close_thunder_cracks.flac",
        "raw":  "thunder_src.flac",
        "wav":  "thunder.wav",
        # The 31 s source has two strikes; the first (5.5-13 s) has
        # the cleanest crack-then-rolling-decay shape. Grab 8 s
        # starting 1 s before the transient.
        "start": 5.5,
        "dur":   8.0,
    },
    "gunshot": {
        # OpenGameArt CC0 zip; we extract mosin.wav (Mosin Nagant rifle
        # outdoor recording) and slice the first shot + decay.
        "url":      "https://opengameart.org/sites/default/files/sounds.zip",
        "raw":      "gunshot_src.zip",
        "zip_entry": "sounds/mosin.wav",
        "extracted": "mosin.wav",
        "wav":      "gunshot.wav",
        "start":    0.0,
        "dur":      3.5,
    },
    "fan": {
        # Freesound's full WAV download is login-gated, but the HQ
        # preview MP3 (320 kbps) is public and the source is CC0, so
        # the format is unrestricted. Sound ID 646877. If this URL
        # ever 404s (Freesound previews are stable but not eternal),
        # fall back to logging in and pulling the original .wav from
        # https://freesound.org/people/Andron827/sounds/646877/
        "url":  "https://cdn.freesound.org/previews/646/646877_1504845-hq.mp3",
        "raw":  "fan_src.mp3",
        "wav":  "fan.wav",
        # Source is ~2 min of steady drone; we grab 8 s starting 10 s
        # in to skip any near-silent lead-in.
        "start": 10.0,
        "dur":   8.0,
    },
    "frogs": {
        # CC0; fetched as HQ MP3 preview (login-free path). Sound ID
        # 503211. Source is 3:08 of a quiet Utah canyon at night.
        # Profiled RMS, picked 50-58 s as the densest chorus.
        "url":  "https://cdn.freesound.org/previews/503/503211_2977885-hq.mp3",
        "raw":  "frogs_src.mp3",
        "wav":  "frogs.wav",
        "start": 50.0,
        "dur":   8.0,
    },
    "sealions": {
        # CC0; HQ MP3 preview. Sound ID 503679. Source is ~8 min of a
        # ~200-animal colony at Crescent City. Profiled RMS, picked
        # 90-98 s as the loudest sustained chorus.
        "url":  "https://cdn.freesound.org/previews/503/503679_9619518-hq.mp3",
        "raw":  "sealions_src.mp3",
        "wav":  "sealions.wav",
        "start": 90.0,
        "dur":   8.0,
    },
}


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        print(f"  already have {path.name}")
        return
    print(f"  downloading {url}")
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "atmospheric-absorption-demo/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        path.write_bytes(r.read())


def detect_peak_db(src: Path, start: float, dur: float) -> float:
    """Return the absolute peak in dBFS of the [start, start+dur] excerpt."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y",
           "-ss", str(start), "-t", str(dur), "-i", str(src),
           "-ac", "1", "-ar", "48000",
           "-af", "volumedetect", "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", res.stderr)
    if not m:
        raise RuntimeError("volumedetect did not return max_volume:\n" + res.stderr)
    return float(m.group(1))


def render(src: Path, out: Path, start: float, dur: float, target_peak_db: float = -6.0) -> None:
    peak = detect_peak_db(src, start, dur)
    gain = target_peak_db - peak
    print(f"  {src.name}: peak={peak:+.2f} dBFS  ->  apply {gain:+.2f} dB  ->  {out.name}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
           "-ss", str(start), "-t", str(dur), "-i", str(src),
           "-ac", "1", "-ar", "48000",
           "-af", f"volume={gain:+.4f}dB",
           "-c:a", "pcm_s16le",
           str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    for kind, spec in SOURCES.items():
        print(f"[{kind}]")
        raw = HERE / spec["raw"]
        wav = HERE / spec["wav"]
        download(spec["url"], raw)
        # Some sources are zip archives; extract the named entry once.
        if "zip_entry" in spec:
            extracted = HERE / spec["extracted"]
            if not extracted.exists() or extracted.stat().st_size < 1000:
                with zipfile.ZipFile(raw) as zf:
                    extracted.write_bytes(zf.read(spec["zip_entry"]))
                print(f"  extracted {spec['zip_entry']} -> {extracted.name}")
            src_for_render = extracted
        else:
            src_for_render = raw
        render(src_for_render, wav, spec["start"], spec["dur"])


if __name__ == "__main__":
    main()
