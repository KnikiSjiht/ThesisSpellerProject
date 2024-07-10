# Expose a few parameters usually accessible via config as a CLI
from dareplane_utils.general.time import sleep_s
from fire import Fire

from mockup_streamer.main import MockupStream


def cli(
    n_channels: int = 32,
    sfreq: int = 512,
    pre_buffer_s: int = 300,
    stream_name: str = "mockup_random",
    markers_t_s: int = 1,
    marker_values: list = ["start_trial"],
):
    """A simple CLI to spawn an LSL mockup stream with random training data

    Parameters
    ----------
    n_channels : int
        number of channels

    sfreq : int
        sampling frequency

    pre_buffer_s : int
        number of samples to be pre-generated, after streaming, another
        set will be generated

    stream_name : str
        name of the stream

    markers_t_s : int
        time interval of markers

    marker_values : list
        values of markers

    """
    cfg = dict(
        sampling_freq=sfreq,
        n_channels=n_channels,
        pre_buffer_s=pre_buffer_s,
        stream_name=stream_name,
        markers=dict(t_interval_s=markers_t_s, values=marker_values),
    )

    streamer = MockupStream(name="test", cfg=cfg)
    dt = 1 / sfreq

    print("=" * 80)
    print(f"Starting stream: {stream_name}")
    print("=" * 80)

    while True:
        sleep_s(dt)
        streamer.push()


if __name__ == "__main__":
    Fire(cli)
