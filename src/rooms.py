"""Demo rooms: a judge screen follows payments made on phones, live.

The judge view creates a room and shows its short code (and a QR of
/app/?code=...). A phone joins with the code and gets a publish token. When the
phone scores a payment with that token, the *server* publishes its own computed
result to the room, so a phone can never inject a decision. The judge reads the
room with its private watch token (read-only).

In memory, one process: the public site runs a single uvicorn worker, and rooms
are short-lived demo state (a restart clears them). Bounded everywhere: room
count, room lifetime, phones per room and events kept per room.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O, 1/I/L confusion
CODE_LENGTH = 6
ROOM_TTL_S = 3 * 3600
MAX_ROOMS = 500
MAX_PHONES = 5
MAX_EVENTS = 100
REPLAY_EVENTS = 20


class RoomError(Exception):
    """A refused room operation; `status` is the HTTP status to report."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Room:
    id: str
    code: str
    watch_token: str
    created: float
    expires: float
    phones: set = field(default_factory=set)       # publish tokens
    events: list = field(default_factory=list)
    seq: int = 0


def normalize_code(code: str) -> str:
    return "".join(c for c in str(code).upper() if c.isalnum())


class Rooms:
    def __init__(self, clock=time.time, ttl_s: int = ROOM_TTL_S, max_rooms: int = MAX_ROOMS):
        self.clock, self.ttl_s, self.max_rooms = clock, ttl_s, max_rooms
        self._rooms: dict[str, Room] = {}
        self._by_code: dict[str, str] = {}
        self._by_publish: dict[str, str] = {}
        self._lock = threading.Lock()

    # -- internals (call with the lock held) ---------------------------------

    def _purge(self) -> None:
        now = self.clock()
        for room in [r for r in self._rooms.values() if r.expires <= now]:
            self._rooms.pop(room.id, None)
            self._by_code.pop(room.code, None)
            for token in room.phones:
                self._by_publish.pop(token, None)

    def _new_code(self) -> str:
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if code not in self._by_code:
                return code

    def _watched(self, room_id: str, watch_token: str) -> Room:
        room = self._rooms.get(room_id)
        if room is None or room.expires <= self.clock():
            raise RoomError(404, "Room not found or expired")
        if not secrets.compare_digest(room.watch_token, str(watch_token)):
            raise RoomError(403, "Wrong watch token")
        return room

    # -- public API -----------------------------------------------------------

    def create(self) -> dict:
        with self._lock:
            self._purge()
            if len(self._rooms) >= self.max_rooms:
                raise RoomError(503, "Too many demo rooms right now; try again later")
            now = self.clock()
            room = Room(id=secrets.token_urlsafe(9), code=self._new_code(),
                        watch_token=secrets.token_urlsafe(24), created=now, expires=now + self.ttl_s)
            self._rooms[room.id] = room
            self._by_code[room.code] = room.id
            return {"room": room.id, "code": room.code, "watch_token": room.watch_token,
                    "expires_in": int(self.ttl_s)}

    def join(self, code: str) -> dict:
        with self._lock:
            self._purge()
            room = self._rooms.get(self._by_code.get(normalize_code(code), ""))
            if room is None:
                raise RoomError(404, "No demo screen with that code (it may have expired)")
            if len(room.phones) >= MAX_PHONES:
                raise RoomError(409, f"This demo screen already has {MAX_PHONES} phones connected")
            token = secrets.token_urlsafe(24)
            room.phones.add(token)
            self._by_publish[token] = room.id
            return {"publish_token": token, "code": room.code,
                    "expires_in": max(0, int(room.expires - self.clock()))}

    def publish(self, publish_token: str, event: dict) -> int:
        """Append a server-built event to the token's room; returns its sequence."""
        with self._lock:
            room = self._rooms.get(self._by_publish.get(str(publish_token), ""))
            if room is None or room.expires <= self.clock():
                raise RoomError(404, "Not connected to a live demo screen")
            room.seq += 1
            room.events.append({**event, "seq": room.seq, "at": self.clock()})
            del room.events[:-MAX_EVENTS]
            return room.seq

    def events(self, room_id: str, watch_token: str, after: int = 0) -> list:
        with self._lock:
            room = self._watched(room_id, watch_token)
            if after <= 0:
                return list(room.events[-REPLAY_EVENTS:])
            return [e for e in room.events if e["seq"] > after]

    def status(self, room_id: str, watch_token: str) -> dict:
        with self._lock:
            room = self._watched(room_id, watch_token)
            return {"room": room.id, "code": room.code, "phones": len(room.phones),
                    "events": room.seq, "expires_in": max(0, int(room.expires - self.clock()))}


class RateLimiter:
    """Per-client token bucket: `rate` requests per minute, bursts up to `rate`."""

    def __init__(self, rate_per_min: int, clock=time.monotonic, max_clients: int = 10_000):
        self.rate, self.clock, self.max_clients = float(rate_per_min), clock, max_clients
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        with self._lock:
            now = self.clock()
            tokens, last = self._buckets.get(client, (self.rate, now))
            tokens = min(self.rate, tokens + (now - last) * self.rate / 60.0)
            if len(self._buckets) >= self.max_clients and client not in self._buckets:
                self._buckets.clear()  # bounded memory; resets everyone (rare)
            if tokens < 1.0:
                self._buckets[client] = (tokens, now)
                return False
            self._buckets[client] = (tokens - 1.0, now)
            return True
