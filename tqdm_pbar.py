from tqdm import tqdm
import time
import datetime


class tqdmFA(tqdm):
    """
    A custom `tqdm` progress bar designed for visualising 
    frame-by-frame generation progress with `matplotlib.animation.FuncAnimation`.
    """

    def __init__(self, *args, **kwargs):
        bar_format_str = (
            "|{bar:25}| {percentage:.1f}% | frame: {n_fmt}/{total_fmt}"
            " | {sec/frame} "
            "[{elapsed}<{remaining}, {frames/sec}]"
        )
        params = {
            "bar_format": bar_format_str,
            "ascii": "-█",
            "colour": "green"   # progress bar colour
        }
        for key, value in params.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)       # pass to constructor of parent class

    @property
    def format_dict(self):
        d = super().format_dict
        d["n_fmt"] = f"{d['n']:,}" if d["n"] else "?"                   # current frame number
        d["total_fmt"] = f"{d['total']:,}" if d["total"] else "?"       # total frames to write (process)
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
        return d


if __name__ == "__main__":

    # --- DUMMY LOOP --- #

    total_frames = 100      # frames to write to video writer (e.g. ffmpeg or pillow)

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