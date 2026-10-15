"""
BBGM player-card Discord bot.

Discord usage:
    pls player FirstName LastName
    pls team Drake           roster + record (region, name, abbrev, or "Drake Bulldogs")
    pls log Drake            this season's results
    pls schedule Drake       remaining games
    pls leaders pts [N]      top N (default 10, max 25) in a stat; `pls leaders` lists stats
    pls load <path or URL>   load an export (.json or gzipped); requires Manage Server
    pls reload               re-read the last loaded source; requires Manage Server

Env vars:
    DISCORD_TOKEN            bot token

The last loaded source is saved to bot_state.json and reloaded on startup.
"""
import asyncio
import difflib
import gzip
import json
import os
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import aiohttp
import discord
import ijson
from discord.ext import commands

STATE_FILE = Path(__file__).with_name("bot_state.json")

CLASS_LABELS = {1: "Fr.", 2: "So.", 3: "Jr.", 4: "Sr."}
STATUS_LABELS = {-1: "Free Agent", -2: "Recruit", -3: "Retired"}
PT_LABELS = {0: "0", 0.75: "-", 1: "", 1.25: "+", 1.75: "++"}  # BBGM playing-time modifier
MAX_MSG = 1900  # stay under Discord's 2000-char limit


def code_blocks(lines):
    """Pack lines into ``` blocks that each fit in one Discord message."""
    chunks, cur, size = [], [], 0
    for ln in lines:
        if cur and size + len(ln) + 1 > MAX_MSG - 10:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        chunks.append(cur)
    return ["```\n" + "\n".join(c) + "\n```" for c in chunks]


def _pg(k):
    return lambda t: t[k] / t["gp"]


def _pct(made, att):
    return lambda t: 100 * t[made] / t[att] if t[att] else 0.0


def _ts(t):
    tsa = 2 * (t["fga"] + 0.44 * t["fta"])
    return 100 * t["pts"] / tsa if tsa else 0.0


# key: (label, value fn, higher_is_better, extra qualifier fn or None, aliases)
# Every leaderboard also requires GP >= half the league's max GP this season.
# Rate stats additionally need 10+ MPG; shooting %s need attempts per game.
MPG10 = lambda t: t["min"] / t["gp"] >= 10
LEADER_STATS = {
    "pts":  ("PPG", _pg("pts"), True, None, ["ppg", "points", "scoring"]),
    "reb":  ("RPG", lambda t: (t["orb"] + t["drb"]) / t["gp"], True, None, ["rpg", "rebounds", "trb"]),
    "ast":  ("APG", _pg("ast"), True, None, ["apg", "assists"]),
    "stl":  ("SPG", _pg("stl"), True, None, ["spg", "steals"]),
    "blk":  ("BPG", _pg("blk"), True, None, ["bpg", "blocks"]),
    "tov":  ("TOV", _pg("tov"), True, None, ["to", "turnovers"]),
    "orb":  ("ORB", _pg("orb"), True, None, ["oreb"]),
    "drb":  ("DRB", _pg("drb"), True, None, ["dreb"]),
    "min":  ("MPG", _pg("min"), True, None, ["mpg", "minutes"]),
    "fga":  ("FGA", _pg("fga"), True, None, []),
    "3pa":  ("3PA", _pg("tpa"), True, None, ["tpa"]),
    "fta":  ("FTA", _pg("fta"), True, None, []),
    "fg%":  ("FG%", _pct("fg", "fga"), True, lambda t: t["fga"] / t["gp"] >= 3, ["fg", "fgpct"]),
    "3p%":  ("3P%", _pct("tp", "tpa"), True, lambda t: t["tpa"] / t["gp"] >= 1.5, ["3p", "tp%", "3pt", "3pct"]),
    "ft%":  ("FT%", _pct("ft", "fta"), True, lambda t: t["fta"] / t["gp"] >= 1, ["ft", "ftpct"]),
    "ts%":  ("TS%", _ts, True, lambda t: (t["fga"] + 0.44 * t["fta"]) / t["gp"] >= 3, ["ts"]),
    "per":  ("PER", lambda t: t["per"], True, MPG10, []),
    "ewa":  ("EWA", lambda t: t["ewa"], True, None, []),
    "ortg": ("ORTG", lambda t: t["ortg"], True, MPG10, ["ortg", "offrtg"]),
    "drtg": ("DRTG", lambda t: t["drtg"], False, MPG10, ["defrtg"]),  # lower is better
}
STAT_ALIASES = {a: k for k, v in LEADER_STATS.items() for a in [k, *v[4]]}

COUNTING = ["gp", "min", "fg", "fga", "tp", "tpa", "ft", "fta",
            "orb", "drb", "ast", "stl", "blk", "tov", "pts", "ewa"]
MIN_WEIGHTED = ["per", "ortg", "drtg"]  # rate stats, weighted by minutes across stints
RATING_KEYS = ["hgt", "stre", "spd", "jmp", "endu", "ins", "dnk", "ft", "fg", "tp",
               "oiq", "diq", "drb", "pss", "reb", "ovr", "pos", "season"]


# ---------------------------------------------------------------- loading

def norm(s: str) -> str:
    """Case/accent/whitespace-insensitive name key."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def _open(path):
    # Sniff gzip magic bytes instead of trusting the extension (URLs often have none).
    with open(path, "rb") as f:
        magic = f.read(2)
    return gzip.open(path, "rb") if magic == b"\x1f\x8b" else open(path, "rb")


def _stream(path, prefix):
    """Stream items under a JSON prefix without loading the whole export."""
    with _open(path) as f:
        for item in ijson.items(f, prefix, use_float=True):
            yield item


def _first(path, prefix, default=None):
    for item in _stream(path, prefix):
        return item
    return default


def aggregate(rows):
    """Combine a season's stat rows (one per team stint) into season totals."""
    t = {k: sum(r.get(k) or 0 for r in rows) for k in COUNTING}
    mins = t["min"]
    for k in MIN_WEIGHTED:
        if mins:
            t[k] = sum((r.get(k) or 0) * (r.get("min") or 0) for r in rows) / mins
        else:
            t[k] = sum(r.get(k) or 0 for r in rows) / len(rows)
    return t


def class_years(reg_seasons, tid, current_season):
    """Years in the program = distinct regular-season stat seasons, plus the
    current season if they're rostered but haven't logged a stat row yet
    (preseason). Swap this out if your dynasty stores class year elsewhere."""
    years = len(reg_seasons)
    if tid >= 0 and current_season and (not reg_seasons or reg_seasons[-1] < current_season):
        years += 1
    return years


def class_label(p):
    if p["tid"] < 0:
        return None
    return CLASS_LABELS.get(p["years"], "Gr." if p["years"] > 4 else None)


def format_record(s):
    rec = f"{int(s.get('won', 0))}-{int(s.get('lost', 0))}"
    for k in ("tied", "otl"):
        if s.get(k):
            rec += f"-{int(s[k])}"
    return rec


def summarize(p, current_season):
    """Reduce a full player object to just what the card needs."""
    ratings = p.get("ratings") or []
    if not ratings:
        return None
    r = max(ratings, key=lambda x: x.get("season", 0))

    reg = [s for s in (p.get("stats") or [])
           if not s.get("playoffs") and s.get("tid", -1) >= 0]
    seasons = sorted({s["season"] for s in reg})
    stat_season = seasons[-1] if seasons else None
    stats = aggregate([s for s in reg if s["season"] == stat_season]) if stat_season else None

    jersey = p.get("jerseyNumber")
    if jersey is None:
        jersey = next((s["jerseyNumber"] for s in reversed(reg) if s.get("jerseyNumber")), None)

    tid = int(p.get("tid", -1))
    born = p.get("born") or {}
    return {
        "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip(),
        "tid": tid,
        "loc": born.get("loc"),
        "college": p.get("college"),
        "hgt": p.get("hgt"),
        "weight": p.get("weight"),
        "jersey": jersey,
        "years": class_years(seasons, tid, current_season),
        "injury": (p.get("injury") or {}).get("gamesRemaining", 0) or 0,
        "pt": p.get("ptModifier", 1),
        "roster_order": p.get("rosterOrder"),
        "ratings": {k: r.get(k) for k in RATING_KEYS},
        "stat_season": stat_season,
        "stats": stats,
    }


class League:
    def __init__(self, path):
        ga = _first(path, "gameAttributes", {})
        if isinstance(ga, list):  # older exports store [{key, value}, ...]
            ga = {a["key"]: a["value"] for a in ga}
        self.season = ga.get("season")

        self.teams, self.records = {}, {}
        for t in _stream(path, "teams.item"):
            tid = int(t["tid"])
            self.teams[tid] = {k: t.get(k) for k in ("region", "name", "abbrev")}
            for s in t.get("seasons") or []:  # older exports nest seasons in teams
                if s.get("season") == self.season:
                    self.records[tid] = format_record(s)
        if not self.records:  # newer exports keep them in a top-level teamSeasons
            for s in _stream(path, "teamSeasons.item"):
                if s.get("season") == self.season:
                    self.records[int(s["tid"])] = format_record(s)

        # Played games: keep only score lines, not box scores.
        self.games = defaultdict(list)
        for g in _stream(path, "games.item"):
            ts = g.get("teams") or []
            if g.get("season") != self.season or len(ts) != 2:
                continue
            rec = {"day": g.get("day"), "gid": g.get("gid", 0),
                   "playoffs": bool(g.get("playoffs")), "ot": int(g.get("overtimes") or 0),
                   "home": int(ts[0]["tid"]), "away": int(ts[1]["tid"]),  # teams[0] is home
                   "pts": (int(ts[0].get("pts") or 0), int(ts[1].get("pts") or 0))}
            for tid in (rec["home"], rec["away"]):
                self.games[tid].append(rec)
        for gs in self.games.values():
            gs.sort(key=lambda g: (g["day"] is None, g["day"] or 0, g["gid"]))

        # Unplayed games.
        self.schedule = defaultdict(list)
        for i, s in enumerate(_stream(path, "schedule.item")):
            home, away = int(s.get("homeTid", -1)), int(s.get("awayTid", -1))
            if home not in self.teams or away not in self.teams:
                continue  # skips placeholders like All-Star games
            rec = {"day": s.get("day"), "order": i, "home": home, "away": away}
            self.schedule[home].append(rec)
            self.schedule[away].append(rec)
        for gs in self.schedule.values():
            gs.sort(key=lambda g: (g["day"] is None, g["day"] or 0, g["order"]))

        self.players = defaultdict(list)
        self.rosters = defaultdict(list)
        self.count = 0
        for p in _stream(path, "players.item"):
            rec = summarize(p, self.season)
            if rec:
                self.players[norm(rec["name"])].append(rec)
                if rec["tid"] >= 0:
                    self.rosters[rec["tid"]].append(rec)
                self.count += 1

    def team_label(self, tid):
        t = self.teams.get(tid) or {}
        return t.get("region") or t.get("name") or t.get("abbrev") or "?"

    # ------------------------------------------------------------ teams

    def team_keys(self):
        """Every way a team can be named -> tid."""
        keys = {}
        for tid, t in self.teams.items():
            region, name, abbrev = t["region"], t["name"], t["abbrev"]
            for k in (abbrev, region, name, f"{region or ''} {name or ''}"):
                if k and k.strip():
                    keys.setdefault(norm(k), set()).add(tid)
        return keys

    def resolve_team(self, query: str):
        """Return (tid, None) or (None, error message)."""
        keys = self.team_keys()
        tids = keys.get(norm(query))
        if not tids:
            close = difflib.get_close_matches(norm(query), keys.keys(), n=3, cutoff=0.7)
            names = sorted({self.team_label(t) for k in close for t in keys[k]})
            hint = f" Did you mean: {', '.join(names)}?" if names else ""
            return None, f"No team named **{query}**.{hint}"
        if len(tids) > 1:
            names = ", ".join(f"{self.team_label(t)} ({self.teams[t]['abbrev']})" for t in sorted(tids))
            return None, f"**{query}** matches several teams: {names}. Try the abbreviation."
        return next(iter(tids)), None

    def team_lookup(self, query: str) -> list[str]:
        tid, err = self.resolve_team(query)
        return [err] if err else self.team_card(tid)

    def team_title(self, tid):
        title = f"{self.season} {self.team_label(tid)}"
        return f"{title}: {self.records[tid]}" if tid in self.records else title

    def opp_label(self, tid):
        rec = self.records.get(tid)
        return f"{self.team_label(tid)} ({rec})" if rec else self.team_label(tid)

    # ------------------------------------------------------------ log / schedule

    def game_log(self, query: str) -> list[str]:
        tid, err = self.resolve_team(query)
        if err:
            return [err]
        games = self.games.get(tid, [])
        if not games:
            return [f"```\n{self.team_title(tid)}\nNo games played yet.\n```"]
        lines, in_playoffs = [self.team_title(tid)], False
        for i, g in enumerate(games, 1):
            if g["playoffs"] and not in_playoffs:
                lines.append("PLAYOFFS")
                in_playoffs = True
            home = g["home"] == tid
            mine, theirs = g["pts"] if home else g["pts"][::-1]
            res = "W" if mine > theirs else "L" if mine < theirs else "T"
            ot = "" if not g["ot"] else " OT" if g["ot"] == 1 else f" {g['ot']}OT"
            opp = g["away"] if home else g["home"]
            n = i  # team's own game number; day only drives ordering
            lines.append(f"Game {n}: {res}, {mine}-{theirs}{ot} "
                         f"{'vs.' if home else '@'} {self.opp_label(opp)}")
        return code_blocks(lines)

    def upcoming(self, query: str) -> list[str]:
        tid, err = self.resolve_team(query)
        if err:
            return [err]
        sched = self.schedule.get(tid, [])
        if not sched:
            return [f"```\n{self.team_title(tid)}\nNo games left on the schedule.\n```"]
        played = len(self.games.get(tid, []))
        lines = [self.team_title(tid)]
        for i, g in enumerate(sched, played + 1):
            home = g["home"] == tid
            opp = g["away"] if home else g["home"]
            n = i  # team's own game number; day only drives ordering
            lines.append(f"Game {n}: {'v' if home else '@'} {self.opp_label(opp)}")
        return code_blocks(lines)

    def team_card(self, tid) -> list[str]:
        title = self.team_title(tid)

        roster = sorted(
            self.rosters.get(tid, []),
            key=lambda p: (p["roster_order"] is None, p["roster_order"] or 0,
                           -(p["ratings"]["ovr"] or 0)),
        )
        if not roster:
            return [f"{title}\nNo players on the roster."]

        rows = []
        for i, p in enumerate(roster, 1):
            r, t = p["ratings"], p["stats"]
            row = [str(i), p["name"], str(p["injury"] or ""), r.get("pos") or "",
                   class_label(p) or "", str(int(r["ovr"] or 0))]
            if t and t["gp"] and p["stat_season"] == self.season:
                gp = t["gp"]
                row += [str(int(gp)), f"{t['min'] / gp:.1f}", f"{t['pts'] / gp:.1f}",
                        f"{(t['orb'] + t['drb']) / gp:.1f}", f"{t['ast'] / gp:.1f}",
                        f"{t['per']:.1f}"]
            else:
                row += ["0", "", "", "", "", ""]
            row.append(PT_LABELS.get(p["pt"], str(p["pt"])))
            rows.append(row)

        header = ["#", "Name", "Inj", "Pos", "Yr", "Ovr", "GP", "Min", "Pts", "Reb", "Ast", "PER", "PT"]
        widths = [max(len(x[c]) for x in rows + [header]) for c in range(len(header))]
        line = lambda cells: " ".join(
            c.ljust(w) if j == len(header) - 1 else c.rjust(w)
            for j, (c, w) in enumerate(zip(cells, widths))
        ).rstrip()
        lines = [line(header)] + [line(r) for r in rows]

        return code_blocks([title] + lines)

    # ------------------------------------------------------------ lookup

    def lookup(self, query: str) -> str:
        matches = self.players.get(norm(query))
        if not matches:
            close = difflib.get_close_matches(norm(query), self.players.keys(), n=3, cutoff=0.75)
            if close:
                names = ", ".join(self.players[k][0]["name"] for k in close)
                return f"No player named **{query}**. Did you mean: {names}?"
            return f"No player named **{query}**."

        # Prefer rostered players, then highest OVR.
        matches = sorted(matches, key=lambda m: (m["tid"] < 0, -(m["ratings"]["ovr"] or 0)))
        card = f"```\n{self.card(matches[0])}\n```"
        if len(matches) > 1:
            card += f"\n_(+{len(matches) - 1} other player(s) with this name)_"
        return card

    # ------------------------------------------------------------ leaders

    def leaders(self, stat: str, n: int = 10) -> str:
        key = STAT_ALIASES.get(stat.lower().strip())
        if not key:
            return f"Unknown stat **{stat}**. Try: {', '.join(LEADER_STATS)}"
        label, fn, desc, extra, _ = LEADER_STATS[key]

        pool = [p for recs in self.rosters.values() for p in recs
                if p["stats"] and p["stats"]["gp"] and p["stat_season"] == self.season]
        if not pool:
            return f"No {self.season} stats yet."
        min_gp = max(p["stats"]["gp"] for p in pool) / 2
        qualified = [p for p in pool
                     if p["stats"]["gp"] >= min_gp and (extra is None or extra(p["stats"]))]
        ranked = sorted(qualified, key=lambda p: fn(p["stats"]), reverse=desc)[:n]
        if not ranked:
            return f"No qualified players for {label}."

        rows = [(str(i), p["name"], self.teams.get(p["tid"], {}).get("abbrev") or self.team_label(p["tid"]),
                 f"{fn(p['stats']):.1f}") for i, p in enumerate(ranked, 1)]
        w = [max(len(r[c]) for r in rows) for c in range(4)]
        body = "\n".join(f"{a.rjust(w[0])}. {b.ljust(w[1])} {c.ljust(w[2])} {d.rjust(w[3])}"
                         for a, b, c, d in rows)
        return f"```\n{self.season} {label} LEADERS (min {int(min_gp)} GP)\n{body}\n```"

    # ------------------------------------------------------------ formatting

    def card(self, p) -> str:
        r, t = p["ratings"], p["stats"]
        f = lambda x: f"{x:.1f}"
        num = lambda x: "?" if x is None else str(int(round(x)))

        if p["tid"] >= 0:
            team = self.team_label(p["tid"])
            cls = class_label(p)
        else:
            team = STATUS_LABELS.get(p["tid"], "?")
            cls = None

        header = [p["name"], f"{num(r['ovr'])} OVR", cls, r.get("pos"), team]
        lines = [" | ".join(x for x in header if x)]
        origin = [x for x in (p["loc"], p["college"]) if x]
        if origin:
            lines.append(" | ".join(origin))
        lines.append("")

        if t and t["gp"]:
            gp = t["gp"]
            pg = lambda k: t[k] / gp
            pct = lambda a, b: 100 * t[a] / t[b] if t[b] else 0.0
            tsa = 2 * (t["fga"] + 0.44 * t["fta"])
            ts = 100 * t["pts"] / tsa if tsa else 0.0
            lines += [
                f"{p['stat_season']} STATS:",
                f"{f(pg('pts'))} PPG | {f(pg('ast'))} APG | {f((t['orb'] + t['drb']) / gp)} RPG | "
                f"{f(pg('stl'))} SPG | {f(pg('blk'))} BPG",
                f"{f(pct('fg', 'fga'))} FG% | {f(pct('tp', 'tpa'))} 3P% | "
                f"{f(pct('ft', 'fta'))} FT% | {f(ts)} TS%",
                f"{f(pg('fga'))} FGA | {f(pg('tpa'))} 3PA | {f(pg('fta'))} FTA",
                f"{f(t['per'])} PER | {f(t['ewa'])} EWA | {f(t['ortg'])} ORTG | {f(t['drtg'])} DRTG",
            ]
        else:
            lines.append(f"{self.season or ''} STATS: no games played".strip())
        lines.append("")

        hgt = p["hgt"]
        size = [f"{num(r['ovr'])} OVR"]
        if hgt:
            hgt = int(hgt)
            size.append(f"{hgt // 12}'{hgt % 12}\"")
        if p["weight"]:
            size.append(f"{num(p['weight'])} LB")
        if p["jersey"] not in (None, ""):
            size.append(str(p["jersey"]))

        g = lambda k: num(r.get(k))
        lines += [
            f"{num(r['season'])} RATINGS:",
            " | ".join(size),
            f"PHYSICAL: {g('hgt')} HGT | {g('stre')} STR | {g('spd')} SPD | {g('jmp')} JMP | {g('endu')} ENDU",
            f"SHOOTING: {g('ins')} INS | {g('dnk')} DNK | {g('ft')} FT | {g('fg')} FG | {g('tp')} 3P",
            f"SKILL: {g('oiq')} OIQ | {g('diq')} DIQ | {g('drb')} DRB | {g('pss')} PSS | {g('reb')} REB",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------- bot

intents = discord.Intents.default()
intents.message_content = True  # also enable "Message Content Intent" in the dev portal

bot = commands.Bot(
    command_prefix=commands.when_mentioned_or("pls ", "Pls ", "PLS "),
    intents=intents,
    case_insensitive=True,
    help_command=None,
    allowed_mentions=discord.AllowedMentions.none(),
)
league: League | None = None
league_source: str | None = None
load_lock = asyncio.Lock()


async def download(url: str) -> str:
    """Stream a URL to a temp file so large exports never sit fully in memory."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        f.write(chunk)
    except Exception:
        os.remove(tmp)
        raise
    return tmp


async def load_source(source: str) -> League:
    """Load from a local path or http(s) URL. The old league keeps serving
    lookups until the new one parses successfully."""
    global league, league_source
    is_url = source.lower().startswith(("http://", "https://"))
    path = await download(source) if is_url else os.path.expanduser(source)
    try:
        new = await asyncio.to_thread(League, path)
    finally:
        if is_url:
            os.remove(path)
    if not new.count:
        raise ValueError("no players found. Is this a BBGM export?")
    league, league_source = new, source
    STATE_FILE.write_text(json.dumps({"source": source}))
    return new


async def run_load(ctx, source: str):
    if load_lock.locked():
        return await ctx.send("A load is already in progress.")
    async with load_lock:
        msg = await ctx.send("Loading export…")
        try:
            new = await load_source(source)
        except Exception as e:
            return await msg.edit(content=f"Load failed: `{type(e).__name__}: {e}`")
        await msg.edit(content=f"Loaded {new.count} players (season {new.season}).")


async def autoload():
    """On startup, reload whatever was last loaded."""
    try:
        source = json.loads(STATE_FILE.read_text())["source"]
    except (FileNotFoundError, KeyError, ValueError):
        return
    async with load_lock:
        try:
            new = await load_source(source)
            print(f"Autoloaded {new.count} players (season {new.season}) from {source}")
        except Exception as e:
            print(f"Autoload of {source} failed: {e}")


async def setup_hook():
    asyncio.create_task(autoload())  # don't block login on a big parse

bot.setup_hook = setup_hook


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.command(name="player")
async def player_cmd(ctx, *, name: str):
    if league is None:
        msg = "League is still loading, try again shortly." if load_lock.locked() \
            else "No league loaded yet. Use `pls load <path or URL>`."
        return await ctx.send(msg)
    await ctx.send(league.lookup(name))


@bot.command(name="team")
async def team_cmd(ctx, *, name: str):
    if league is None:
        msg = "League is still loading, try again shortly." if load_lock.locked() \
            else "No league loaded yet. Use `pls load <path or URL>`."
        return await ctx.send(msg)
    for chunk in league.team_lookup(name):
        await ctx.send(chunk)


def not_ready():
    if league is not None:
        return None
    return ("League is still loading, try again shortly." if load_lock.locked()
            else "No league loaded yet. Use `pls load <path or URL>`.")


@bot.command(name="log", aliases=["results"])
async def log_cmd(ctx, *, name: str):
    if msg := not_ready():
        return await ctx.send(msg)
    for chunk in league.game_log(name):
        await ctx.send(chunk)


@bot.command(name="schedule", aliases=["sched", "upcoming"])
async def schedule_cmd(ctx, *, name: str):
    if msg := not_ready():
        return await ctx.send(msg)
    for chunk in league.upcoming(name):
        await ctx.send(chunk)


@bot.command(name="leaders", aliases=["leader", "top"])
async def leaders_cmd(ctx, stat: str = None, n: int = 10):
    if league is None:
        msg = "League is still loading, try again shortly." if load_lock.locked() \
            else "No league loaded yet. Use `pls load <path or URL>`."
        return await ctx.send(msg)
    if stat is None:
        return await ctx.send("Usage: `pls leaders <stat> [count]`\nStats: " + ", ".join(LEADER_STATS))
    await ctx.send(league.leaders(stat, max(1, min(n, 25))))


HELP_TEXT = """**BBGM Bot commands**
`pls player <First Last>`: player card with this season's stats and ratings
`pls team <name>`: record and roster (region, nickname, or abbreviation)
`pls log <team>`: this season's results
`pls schedule <team>`: remaining games
`pls leaders <stat> [count]`: top players in a stat (default 10, max 25)
  Stats: {stats}
`pls help`: this list

**Admin** (requires Manage Server)
`pls load <path or URL>`: load a BBGM export (.json or gzipped)
`pls reload`: re-read the last loaded export
{status}"""


@bot.command(name="help", aliases=["commands"])
async def help_cmd(ctx):
    if league:
        status = f"\n_Loaded: season {league.season}, {league.count} players._"
    else:
        status = "\n_No league loaded yet._"
    await ctx.send(HELP_TEXT.format(stats=", ".join(LEADER_STATS), status=status))


@bot.command(name="load")
@commands.has_permissions(manage_guild=True)
async def load_cmd(ctx, *, source: str):
    # Tolerate <url> (Discord embed suppression) and quoted paths.
    await run_load(ctx, source.strip().strip("<>\"'"))


@bot.command(name="reload")
@commands.has_permissions(manage_guild=True)
async def reload_cmd(ctx):
    if not league_source:
        return await ctx.send("Nothing loaded yet. Use `pls load <path or URL>`.")
    await run_load(ctx, league_source)


USAGE = {
    "player": "pls player FirstName LastName",
    "team": "pls team TeamName",
    "leaders": "pls leaders <stat> [count]",
    "log": "pls log TeamName",
    "schedule": "pls schedule TeamName",
    "load": "pls load <path or URL>",
}


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return  # "pls" is shared with other bots; stay quiet
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Usage: `{USAGE.get(ctx.command.name, ctx.command.name)}`")
    elif isinstance(error, commands.BadArgument):
        await ctx.send(f"Usage: `{USAGE.get(ctx.command.name, ctx.command.name)}`")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You need Manage Server to load exports.")
    else:
        raise error


if __name__ == "__main__":
    bot.run(os.environ["DISCORD_TOKEN"])