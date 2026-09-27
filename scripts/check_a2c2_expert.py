"""Exploratory check (a), step 1 (docs/results/b2b5-checks.md): peek into the A2C2 expert data (gs://rtc-assets) by
HTTP range reads, a few MB per level: the npz members, the expert's action ranges per dimension and the done rate over
the first T steps of every env (a2c2_expert.txt); with `solved`, the expert's solved share over all episodes of all 12
levels from done.npy and solved.npy only, ~2 MB per level (a2c2_expert_solved.txt).
    uv run --offline python scripts/check_a2c2_expert.py [solved]"""
import io
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np

URL = "https://storage.googleapis.com/rtc-assets/expert/data/worlds_l_{}.npz"
LEVELS = ("trampoline", "mjc_walker", "car_launch")
T = 64  # time steps read from the start of each member (data are [steps, envs, ...])


class Ranged(io.RawIOBase):
    """A read-only seekable file over HTTP range requests; counts the bytes fetched."""

    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def head_rows(zf, name, rows):
    """The first `rows` rows of a (possibly deflated) .npy member, reading only as far as needed."""
    with zf.open(name) as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        per_row = int(np.prod(shape[1:])) * dtype.itemsize
        n = min(rows, shape[0])
        buf = f.read(n * per_row)
    return shape, np.frombuffer(buf, dtype).reshape((n, *shape[1:]))


def main():
    lines = []
    for lv in LEVELS:
        src = Ranged(URL.format(lv))
        zf = zipfile.ZipFile(io.BufferedReader(src, buffer_size=1 << 20))
        members = {i.filename: (i.file_size, i.compress_size, i.compress_type) for i in zf.infolist()}
        lines.append(f"== {lv}: {src.size / 1e9:.2f} GB; members {members}")
        shape_a, act = head_rows(zf, "action.npy", T)
        shape_d, done = head_rows(zf, "done.npy", T)
        a = act.reshape(-1, act.shape[-1]).astype(float)
        q = np.percentile(a, [0, 1, 50, 99, 100], axis=0).round(2)
        lines.append(f"action {shape_a}, first {T} steps: per-dim min/p1/median/p99/max\n{q}")
        lines.append(f"  share |a| > 1 per dim: {(np.abs(a) > 1).mean(0).round(3)}; mean {a.mean(0).round(2)}")
        lines.append(f"done {shape_d}: rate in first {T} steps {done.mean():.4f}; envs with a done {done.any(0).mean():.3f}")
        for extra in sorted(set(members) - {"obs.npy", "action.npy", "done.npy"}):
            shp, x = head_rows(zf, extra, T)
            lines.append(f"{extra} {shp}: first {T} steps mean {np.asarray(x, float).mean():.4f} max {np.asarray(x, float).max():.4f}")
        lines.append(f"  fetched {src.fetched / 1e6:.1f} MB")
        print("\n".join(lines[-6:]), flush=True)
    out = pathlib.Path("results/b2b5/checks/a2c2_expert.txt")
    out.write_text("\n".join(lines) + "\n")


def solved():
    lines = []
    for lv in ("grasp_easy", "catapult", "cartpole_thrust", "hard_lunar_lander", "mjc_half_cheetah", "mjc_swimmer",
               "mjc_walker", "h17_unicycle", "chain_lander", "catcher_v3", "trampoline", "car_launch"):
        src = Ranged(URL.format(lv))
        zf = zipfile.ZipFile(io.BufferedReader(src, buffer_size=1 << 20))
        with zf.open("done.npy") as f:
            ends = np.load(f).astype(bool)
        with zf.open("solved.npy") as f:
            ok = np.load(f)
        lines.append(f"{lv}: episodes {ends.sum()}, solved share {ok[ends].mean():.3f} fetched {src.fetched / 1e6:.1f} MB")
        print(lines[-1], flush=True)
    pathlib.Path("results/b2b5/checks/a2c2_expert_solved.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    solved() if sys.argv[1:] == ["solved"] else main()
