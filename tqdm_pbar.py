from tqdm import tqdm
import time
import datetime


class tqdmFA(tqdm):
    """
    A custom `tqdm` progress bar designed for visualising 
    frame-by-frame generation progress with `matplotlib.animation.FuncAnimation`.

    Args:
        fps (int): Frames per second of the video file.

    `|███████------------------| 30.7% | frame: 168/548 | video duration: 0:02/0:09 | 40 ms/frame [00:06<00:15, 25.3 frames/sec]`
    """

    def __init__(self, *args, fps: int = 60, **kwargs):
        bar_format_str = (
            "|{bar:30}| {percentage:.1f}%"
            " | frame: {n_fmt}/{total_fmt}"
            " | video duration: {vid_dur}"
            " | {sec/frame} "
            "[{elapsed}<{remaining}, {frames/sec}]"
        )
        params = {
            "bar_format": bar_format_str,
            "ascii": "-█",
            "colour": "green"   # progress bar colour
        }
        self.fps = fps      # frames per second of MP4 video file
        for key, value in params.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)       # pass to constructor of parent class

    @property
    def format_dict(self) -> dict:
        """Override the format_dict property to include custom formatting for video writing progress."""
        d = super().format_dict
        d["n_fmt"] = f"{d['n']:,}" if d["n"] else "?"                   # current frame number
        d["total_fmt"] = f"{d['total']:,}" if d["total"] else "?"       # total frames to write (process)
        # written frames/sec and time per frame:
        if d["rate"]:
            rate = d["rate"]
            # time taken to write a frame:
            if 1 / rate < 1:
                d["sec/frame"] = f"{1 / rate * 1e3:.0f} ms/frame"       # in milliseconds if fast
            else:
                d["sec/frame"] = f"{1 / rate:.2f} sec/frame"            # normal seconds
            # no. of frames written per second:
            d["frames/sec"] = f"{rate:.1f} frames/sec"
        else:
            d["sec/frame"] = "? sec/frame"
            d["frames/sec"] = "? frames/sec"
        # written video duration in seconds:
        def fmt(t) -> None:
            """Format timedelta as M:SS (not H:MM:SS)"""
            minutes, seconds = divmod(t.seconds, 60)
            return f"{minutes}:{seconds:02}"
        if self.n and self.total:
            elapsed = datetime.timedelta(seconds=self.n/self.fps)         # total steps / FPS
            duration = datetime.timedelta(seconds=self.total/self.fps)    # total video duration
            d["vid_dur"] = f"{fmt(elapsed)}/{fmt(duration)}"
        else:
            d["vid_dur"] = f"?:??/?:??"
        return d


if __name__ == "__main__":

    # --- DUMMY LOOP --- #

    total_frames = 548      # frames to write to video writer (e.g. ffmpeg or pillow)

    pbar = tqdmFA(iterable=range(total_frames))
    for _ in pbar:
        pbar.update(1)
        time.sleep(0.0432)    # dummy delay
    pbar.close()

    # alternative usage:
    # with tqdmFA(total=total_frames) as pbar:
    #     for _ in range(total_frames):
    #         pbar.update(1)
    #         time.sleep(0.05)

    # --- ADDITIONAL STATS --- #

    t = datetime.timedelta(seconds=int(pbar.format_dict["elapsed"]))    # total elapsed time
    print(f"\n\ntotal elapsed time: {t}")
    avg_iter_per_sec = total_frames / t.total_seconds()
    if 1 / avg_iter_per_sec < 1:
        avg_rate = f"{1 / avg_iter_per_sec * 1e3:.0f} ms/frame"
    else:
        avg_rate = f"{1 / avg_iter_per_sec:.2f} sec/frame"
    print(f"{avg_iter_per_sec:.1f} frames/sec processed ({avg_rate})")