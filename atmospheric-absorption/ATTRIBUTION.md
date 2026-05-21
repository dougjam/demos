# Audio attribution

## Live-generated (Web Audio AudioWorklet)

- **White noise.** Gaussian samples (Box-Muller transform), scaled to
  approximately -12 dBFS.
- **Pink noise.** Paul Kellet's IIR pink-noise filter, fed by uniform
  white noise.

## File-based presets (`thunder`, `gunshot`, `voice`, `music`, `fan`)

Downloaded and rendered by
[`audio/fetch_real_audio.py`](audio/fetch_real_audio.py). Each clip is
re-encoded to mono 48 kHz / 16-bit PCM, sliced from the source as noted
below, and peak-normalized to -6 dBFS. Licenses are a mix of public
domain (pre-1926 US recordings and US federal works), CC0, and CC BY-SA.

| Output        | Source                                                                 | Origin                                                                                                                       | License                          |
|---------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| `thunder.wav` (8 s) | "Very close thunder cracks" by [Delorean76](https://commons.wikimedia.org/wiki/User:Delorean76) — sharp transient with natural rolling decay; we use the 5.5–13.5 s segment of the 31 s source. | Wikimedia Commons — [`File:Very_close_thunder_cracks.flac`](https://commons.wikimedia.org/wiki/File:Very_close_thunder_cracks.flac) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) — © Delorean76, used with attribution; modified (trimmed, downsampled, peak-normalized) |
| `gunshot.wav` (3.5 s) | "Gunshot Sounds" pack by Tabasco — outdoor target-range recordings of CZ-52 pistol, Mosin Nagant rifle, SKS, and shotgun. We use the first shot from `mosin.wav` (Mosin Nagant 7.62 mm rifle). | OpenGameArt.org — ["Gunshot Sounds"](https://opengameart.org/content/gunshot-sounds) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (no rights reserved) |
| `voice.wav` (9 s) | Neil Armstrong, *"That's one small step for a man, one giant leap for mankind"* (Apollo 11, July 20, 1969) — quarter-million-mile S-band radio carries strong band-limited / loudspeaker character. We use the 14.8–23.8 s segment of the 24 s source. | Wikimedia Commons — [`File:Armstrong_Small_Step.ogg`](https://commons.wikimedia.org/wiki/File:Armstrong_Small_Step.ogg) | Public domain (NASA — US federal work) |
| `music.wav` (7 s) | Original Dixieland Jass Band, *Tiger Rag* (1918) — full jazz band with cornet, clarinet, trombone, piano, drums. We use the 30–37 s segment of the source. | Wikimedia Commons — [`File:Tiger_Rag_ODJB.ogg`](https://commons.wikimedia.org/wiki/File:Tiger_Rag_ODJB.ogg) | Public domain (US — sound recordings published before 1926) |
| `fan.wav` (8 s) | *"data center ambient noise"* by [Andron827](https://freesound.org/people/Andron827/) — close-mic'd recording inside a computer test lab; constant broadband drone, no transients. We use the 10–18 s segment of the ~2 min source. | Freesound — [sound 646877](https://freesound.org/people/Andron827/sounds/646877/). The original WAV download requires a Freesound login, so we fetch the publicly hosted 320 kbps HQ MP3 preview and transcode. | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (no rights reserved — attribution provided as a courtesy) |

The downloaded source files (`thunder_src.flac`, `gunshot_src.zip` +
extracted `mosin.wav`, `voice_src.ogg`, `music_src.ogg`, `fan_src.mp3`)
are kept alongside the rendered WAVs so the rendering is reproducible
without re-downloading.

## Synthesized

All four preset WAVs now come from real public-domain or CC0 recordings.
[`audio/generate_audio.py`](audio/generate_audio.py) is kept in the
repository as a reference but is no longer used by default; uncomment
its preset entries to regenerate purely synthetic substitutes.

## Code

The demo code and Python reference are (c) 2026 Doug James, Stanford
University, BSD-2-Clause.

The Chart.js library is loaded from a CDN under MIT license.

## Adding your own clips

If you replace any WAV, ensure the substitute is CC0, public domain,
own-recorded, or otherwise license-compatible with the BSD-2-Clause
license that governs the rest of this demo, and update this file with
attribution.
