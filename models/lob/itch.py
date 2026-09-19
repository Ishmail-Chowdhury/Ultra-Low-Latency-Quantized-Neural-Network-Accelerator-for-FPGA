"""Strict single-symbol displayed-book reconstruction from Nasdaq ITCH 5.0.

Input must start at the day's Start of Messages event, not at an arbitrary
offset. Non-displayed trade prints never change the displayed order book.
"""

from collections import Counter

from models.lob.data import Snapshot


LENGTHS = dict(S=12, R=39, H=25, Y=20, L=26, V=35, W=12, K=28, J=35,
               h=21, A=36, F=40, E=31, C=36, X=23, D=19, U=35,
               P=44, Q=40, B=19, I=50, N=20)


def messages(stream):
    """Read BinaryFILE's two-byte big-endian length followed by an ITCH message."""
    while header := stream.read(2):
        if len(header) != 2:
            raise ValueError("Truncated ITCH frame header")
        length = int.from_bytes(header, "big")
        if length == 0:
            if stream.read(1):
                raise ValueError("Data after BinaryFILE session terminator")
            return
        if not 11 <= length <= 100:
            raise ValueError(f"Invalid ITCH message length {length}")
        body = stream.read(length)
        if len(body) != length:
            raise ValueError("Truncated ITCH frame body")
        kind = chr(body[0])
        if kind not in LENGTHS or length != LENGTHS[kind]:
            raise ValueError(f"Unsupported ITCH type or length: {kind!r}/{length}")
        yield body


def uint(body, start, length):
    return int.from_bytes(body[start:start + length], "big")


class ItchBook:
    def __init__(self, symbol):
        if not symbol or len(symbol) > 8 or not symbol.isascii():
            raise ValueError("ITCH symbol must contain 1..8 ASCII characters")
        self.symbol = symbol
        self.locate = None
        self.started = False
        self.ended = False
        self.trading = False
        self.orders = {}
        self.depth = {"B": Counter(), "S": Counter()}
        self.counts = Counter()
        self.last_timestamp = -1

    def add(self, reference, side, price, shares):
        if reference in self.orders or side not in self.depth or price <= 0 or shares <= 0:
            raise ValueError("Invalid or duplicate ITCH add order")
        self.orders[reference] = (side, price, shares)
        self.depth[side][price] += shares

    def reduce(self, reference, shares=None):
        if reference not in self.orders:
            raise ValueError("ITCH update references an unknown order; source may have gaps")
        side, price, remaining = self.orders[reference]
        shares = remaining if shares is None else shares
        if not 0 < shares <= remaining:
            raise ValueError("ITCH reduction exceeds remaining shares or is nonpositive")
        self.depth[side][price] -= shares
        if self.depth[side][price] == 0:
            del self.depth[side][price]
        if remaining == shares:
            del self.orders[reference]
        else:
            self.orders[reference] = (side, price, remaining - shares)
        return side

    def apply(self, body):
        """Return (timestamp, book_changed); framing is validated by messages()."""
        kind = chr(body[0])
        if kind not in LENGTHS or len(body) != LENGTHS[kind]:
            raise ValueError("Unsupported ITCH type or length")
        timestamp = uint(body, 5, 6)
        if timestamp >= 86400 * 10**9:
            raise ValueError("ITCH timestamp outside day")
        self.counts[kind] += 1
        if not self.started:
            if kind != "S" or body[11:12] != b"O":
                raise ValueError("ITCH source must begin with Start of Messages")
            self.started = True
            return timestamp, False
        if self.ended:
            raise ValueError("ITCH messages after end of day")
        if kind == "S":
            if body[11:12] == b"O":
                raise ValueError("Multiple ITCH days in one file")
            self.ended = body[11:12] == b"C"
            return timestamp, False
        locate = uint(body, 1, 2)
        if kind == "R" and body[11:19].decode("ascii").strip() == self.symbol:
            if self.locate is not None:
                raise ValueError("Duplicate symbol directory entry")
            self.locate = locate
            return timestamp, False
        if self.locate is None or locate != self.locate:
            return timestamp, False
        if timestamp < self.last_timestamp:
            raise ValueError("Target-symbol ITCH timestamps decrease")
        self.last_timestamp = timestamp
        if kind == "H":
            self.trading = body[19:20] == b"T"
            return timestamp, True  # A halt must break the exported session.
        if kind in ("A", "F"):
            if body[24:32].decode("ascii").strip() != self.symbol:
                raise ValueError("Symbol differs from stock locate directory")
            self.add(uint(body, 11, 8), chr(body[19]), uint(body, 32, 4), uint(body, 20, 4))
        elif kind in ("E", "C", "X"):
            self.reduce(uint(body, 11, 8), uint(body, 19, 4))
        elif kind == "D":
            self.reduce(uint(body, 11, 8))
        elif kind == "U":
            old, new = uint(body, 11, 8), uint(body, 19, 8)
            shares, price = uint(body, 27, 4), uint(body, 31, 4)
            if new in self.orders or shares <= 0 or price <= 0:
                raise ValueError("Invalid ITCH replacement")
            side = self.reduce(old)
            self.add(new, side, price, shares)
        else:
            # P/Q/B are trade reports, not changes to displayed depth.
            return timestamp, False
        return timestamp, True

    def snapshot(self, timestamp, session, levels=5):
        bids = sorted(self.depth["B"], reverse=True)[:levels]
        asks = sorted(self.depth["S"])[:levels]
        if not self.trading or len(bids) < levels or len(asks) < levels or bids[0] >= asks[0]:
            return None
        return Snapshot(timestamp, session, self.symbol,
                        tuple(p / 10000 for p in bids), tuple(self.depth["B"][p] for p in bids),
                        tuple(p / 10000 for p in asks), tuple(self.depth["S"][p] for p in asks))
