from minispeaker.devices import default_speaker as default_speaker, available_speakers as available_speakers
from minispeaker.player import Speakers as Speakers
from minispeaker.tracks import Track as Track # NOTE: This is exposed for type-hinting purposes - refactoring work will be required to support direct instantiation by end users with ease of use

def main() -> None:
    print("Hello from pyminispeaker!")
