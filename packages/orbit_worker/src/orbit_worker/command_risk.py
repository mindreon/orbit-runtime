"""How risky a Bash command is: pure functions, no I/O.

`classify(command)` is `low`, `medium`, `high` or `catastrophic`; `assess(command)` also says what the command writes
and whether it only runs skill scripts. A command is read the way a shell reads it: quotes and backslashes are resolved
(`r\\m`, `"r"m` and `rm` are one word), compound commands (`&&`, `||`, `;`, `|`, `&`, newlines, subshells, groups, loops)
are split, `$(...)`, backticks, `<(...)` and `${...}` are read as commands of their own, and the worst part decides. A
program the classifier does not know is `medium`; one it cannot even name (`$CMD`) is `high`.

    low           read-only: ls, cat, grep, find without -delete/-exec, git status, echo, ...
    medium        everything else that is not listed below: builds, installs into the workspace, git commit, python x.py
    high          privilege, recursive forced delete, code piped into a shell, global installs, writes to dotfiles/.ssh/.git,
                  eval, uploads, kill -9, force push, ...
    catastrophic  what no preset allows: rm -rf on / or ~ or a system dir, mkfs, dd onto a disk, fork bombs, shutdown,
                  chmod/chown -R on /, writes to a disk device

The tables reuse AgentScope's own lists (`agentscope.tool._constants`) for dangerous files and directories; the shell syntax
is read here, without a parser dependency, because a wrong answer must lean to the riskier side and never raise.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Literal

from agentscope.tool._constants import DEFAULT_DANGEROUS_DIRECTORIES, DEFAULT_DANGEROUS_FILES

Level = Literal["low", "medium", "high", "catastrophic"]
_RANK: dict[Level, int] = {"low": 0, "medium": 1, "high": 2, "catastrophic": 3}

# Where skills are staged in the sandbox (`sandbox.SKILLS_DIR` under `workspace.WORKSPACE_DIR`). Spelled out here so the
# classifier stays free of the workspace and Temporal imports.
WORKSPACE = "/workspace"
SKILLS_PREFIX = f"{WORKSPACE}/.skills/"
_MAX_DEPTH = 6


@dataclass(frozen=True)
class Assessment:
    level: Level = "low"
    # Why the level is what it is, as a short command-like phrase; empty for `low`.
    reason: str = ""
    # The command is known to write files (a redirect, cp, mkdir, sed -i, git commit, ...). Unknown programs are not.
    writes: bool = False
    # It writes to a file or directory AgentScope guards (.bashrc, .ssh, .git, .env, ...).
    protected_write: bool = False
    # Every part of it that is not read-only runs a script under /workspace/.skills/, and at least one does.
    skill_script: bool = False


def is_protected_path(path: str) -> bool:
    """Whether `path` is a file or directory AgentScope guards (shell start-up files, .ssh, .git, .env, credentials)."""
    return bool(path) and _protected(_resolve(path, WORKSPACE))


def classify(command: str) -> Level:
    return assess(command).level


def assess(command: str) -> Assessment:
    acc = _Acc()
    _walk(command, acc, depth=0, cwd=WORKSPACE)
    return Assessment(
        level=acc.level,
        reason=acc.reason,
        writes=acc.writes,
        protected_write=acc.protected,
        skill_script=acc.skills_seen and acc.skills_ok and _RANK[acc.level] < _RANK["high"],
    )


# ---------------------------------------------------------------------------------------------------------------------
# Tables

_PROTECTED_FILES = frozenset(
    name.lower()
    for name in (
        *DEFAULT_DANGEROUS_FILES,
        # Credential and shell start-up files AgentScope's list does not name.
        ".bash_logout", ".zshenv", ".zlogin", ".inputrc", ".pgpass", ".git-credentials", "id_rsa", "id_ed25519",
        "id_ecdsa", "known_hosts", "credentials",
    )
)
_PROTECTED_DIRS = frozenset(
    name.lower() for name in (*DEFAULT_DANGEROUS_DIRECTORIES, ".aws", ".gnupg", ".kube", ".docker")
)
# Removing, moving or re-owning these (or what is directly inside the first group) is catastrophic.
_SYSTEM_STRICT = frozenset({"/bin", "/boot", "/dev", "/etc", "/lib", "/lib32", "/lib64", "/proc", "/root", "/sbin", "/sys", "/usr"})
_SYSTEM_LOOSE = frozenset({"/home", "/media", "/mnt", "/opt", "/run", "/srv", "/var"})
_SYSTEM_WRITE = _SYSTEM_STRICT | {"/var", "/opt"}
_BLOCK_DEVICE = re.compile(r"^/dev/(sd[a-z]|hd[a-z]|vd[a-z]|xvd[a-z]|nvme\d|mmcblk\d|loop\d|dm-\d|md\d|disk\d|sr\d|mapper/)")
_HARMLESS_DEVICES = frozenset({"/dev/null", "/dev/zero", "/dev/stdout", "/dev/stderr", "/dev/stdin", "/dev/tty", "/dev/full"})
_FORK_BOMB = re.compile(r"([A-Za-z_][\w-]*|:)\s*\(\s*\)\s*\{[^}]*\b\1\b\s*\|\s*&?\s*\1\b|:\s*\|\s*:\s*&")
_FETCH = re.compile(r"\b(curl|wget|nc|ncat|netcat|socat|fetch)\b")

_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "ash"})
_INTERPRETERS = frozenset({"python", "perl", "ruby", "node", "php", "lua", "deno", "bun", "tclsh", "Rscript"})
_DISK_FORMAT = frozenset({"mke2fs", "mkswap", "wipefs", "mkfs"})
_POWER = frozenset({"shutdown", "reboot", "halt", "poweroff"})
_HIGH_PROGRAMS = {
    "doas": "doas", "pkexec": "pkexec", "mount": "mount", "umount": "umount", "swapon": "swapon", "swapoff": "swapoff",
    "losetup": "losetup", "modprobe": "modprobe", "insmod": "insmod", "rmmod": "rmmod", "iptables": "iptables",
    "ip6tables": "ip6tables", "nft": "nft", "ufw": "ufw", "firewall-cmd": "firewall-cmd", "useradd": "useradd",
    "userdel": "userdel", "usermod": "usermod", "groupadd": "groupadd", "passwd": "passwd", "chpasswd": "chpasswd",
    "visudo": "visudo", "chsh": "chsh", "at": "at", "fdisk": "fdisk", "sfdisk": "sfdisk", "cfdisk": "cfdisk",
    "parted": "parted", "gdisk": "gdisk", "sgdisk": "sgdisk", "nc": "nc (network)", "ncat": "ncat (network)",
    "netcat": "netcat (network)", "socat": "socat (network)", "telnet": "telnet (network)", "ssh": "ssh (remote)",
    "scp": "scp (remote)", "sftp": "sftp (remote)", "ftp": "ftp (remote)", "killall": "killall", "pkill": "pkill",
    "kexec": "kexec",
}
_LOW_PROGRAMS = frozenset(
    ["ls", "ll", "la", "dir", "cat", "head", "tail", "less", "more", "file", "stat", "wc", "grep", "egrep", "fgrep", "rg", "ag", "ack", "tree", "pwd", "which", "whereis", "type", "echo", "printf", "true", "false", "whoami", "id", "uname", "basename", "dirname", "realpath", "readlink", "du", "df", "free", "uptime", "ps", "top", "sort", "uniq", "cut", "tr", "diff", "cmp", "comm", "tac", "nl", "rev", "seq", "sleep", "test", "[", "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum", "od", "hexdump", "strings", "jq", "column", "fold", "fmt", "expand", "unexpand", "cd", "pushd", "popd", ":", "export", "unset", "read", "wait", "exit", "return", "shift", "local", "declare", "readonly", "alias", "unalias", "history", "set", "shopt", "ulimit", "umask", "getconf", "nproc", "lscpu", "lsblk", "locale", "base64", "printenv_", "expr", "bc", "tput", "clear", "dos2unix_", "pgrep", "lsof_", "nslookup", "dig", "host", "whatis", "man", "info", "help", "tty", "groups", "logname", "arch", "date"]
)
_GIT_LOW = frozenset(
    ["status", "log", "diff", "show", "ls-files", "ls-tree", "rev-parse", "rev-list", "describe", "shortlog", "blame", "cat-file", "grep", "diff-tree", "name-rev", "check-ignore", "count-objects", "whatchanged", "help", "version", "var", "fsck", "verify-commit"]
)
_NO_WRITE_GIT = _GIT_LOW | {"push", "ls-remote"}
_WRITE_PROGRAMS = frozenset(
    ["rm", "rmdir", "mv", "cp", "mkdir", "touch", "tee", "ln", "chmod", "chown", "chgrp", "truncate", "dd", "install", "patch", "shred", "rsync", "unzip", "gzip", "gunzip", "bzip2", "xz", "zip", "tar_", "unlink", "mkfifo", "mknod"]
)
_PKG_SYSTEM = frozenset({"apt", "apt-get", "aptitude", "dpkg", "yum", "dnf", "apk", "pacman", "zypper", "brew", "snap", "port", "rpm"})
_PKG_READ = frozenset(
    {"list", "search", "show", "info", "policy", "query", "-l", "-L", "-s", "-S", "--list", "--status", "-q", "-qa", "-qi", "outdated", "--version"}
)
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Words that open or close a compound command: what follows them is a command, or nothing.
_LEAD_KEYWORDS = frozenset({"!", "{", "}", "then", "do", "else", "elif", "if", "while", "until", "time", "coproc"})
_EMPTY_KEYWORDS = frozenset({"for", "select", "case", "done", "fi", "esac", "function", "in"})


class _Acc:
    def __init__(self) -> None:
        self.level: Level = "low"
        self.reason = ""
        self.writes = False
        self.protected = False
        self.skills_seen = False
        self.skills_ok = True

    def bump(self, level: Level, reason: str) -> None:
        if _RANK[level] > _RANK[self.level]:
            self.level = level
            self.reason = reason

    def merge(self, other: _Acc) -> None:
        self.bump(other.level, other.reason)
        self.writes = self.writes or other.writes
        self.protected = self.protected or other.protected


# ---------------------------------------------------------------------------------------------------------------------
# Lexer: shell text to simple commands


@dataclass
class _Cmd:
    words: list[str]
    redirects: list[tuple[str, str]]
    subs: list[str]
    piped_in: bool = False


class _Lexer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.cmds: list[_Cmd] = []
        self.cur = _Cmd([], [], [])
        self.word: list[str] = []
        self.in_word = False
        self.quoted = False
        self.redirect: str | None = None
        self.heredocs: list[tuple[str, bool, bool, _Cmd]] = []

    def end_word(self) -> None:
        if not self.in_word:
            return
        word = "".join(self.word)
        self.word, self.in_word = [], False
        quoted, self.quoted = self.quoted, False
        op, self.redirect = self.redirect, None
        if op is None:
            self.cur.words.append(word)
        elif op in ("<<", "<<-"):
            self.heredocs.append((word, quoted, op == "<<-", self.cur))
        elif op != "dup":
            self.cur.redirects.append((op, word))

    def end_cmd(self, operator: str = ";") -> None:
        self.end_word()
        self.redirect = None
        if self.cur.words or self.cur.redirects or self.cur.subs:
            self.cmds.append(self.cur)
        self.cur = _Cmd([], [], [], piped_in=operator in ("|", "|&"))

    def add(self, text: str) -> None:
        self.word.append(text)
        self.in_word = True

    def run(self) -> list[_Cmd]:
        text, n, i = self.text, len(self.text), 0
        while i < n:
            c = text[i]
            if c in " \t":
                self.end_word()
                i += 1
            elif c == "\n":
                self.end_cmd()
                i = self.read_heredocs(i + 1)
            elif c == "\\":
                if i + 1 < n and text[i + 1] == "\n":
                    i += 2
                elif i + 1 < n:
                    self.add(text[i + 1])
                    i += 2
                else:
                    i += 1
            elif c == "'":
                end = text.find("'", i + 1)
                end = n if end < 0 else end
                self.add(text[i + 1 : end])
                self.quoted = True
                i = end + 1
            elif c == '"':
                i = self.double_quoted(i + 1)
                self.quoted = True
            elif c == "`":
                i = self.backtick(i)
            elif c == "$":
                i = self.dollar(i)
            elif c in "<>" and i + 1 < n and text[i + 1] == "(":
                end = _balanced(text, i + 2)
                self.cur.subs.append(text[i + 2 : end])
                self.add("$()")
                i = end + 1
            elif c in "<>" or (c == "&" and text[i : i + 2] == "&>"):
                i = self.redirection(i)
            elif c in ";&|":
                i = self.separator(i)
            elif c in "()":
                self.end_cmd()
                i += 1
            else:
                self.add(c)
                i += 1
        self.end_cmd()
        return self.cmds

    def separator(self, i: int) -> int:
        two = self.text[i : i + 2]
        if two in ("&&", "||", ";;", "|&"):
            self.end_cmd(two)
            return i + 2
        self.end_cmd(self.text[i])
        return i + 1

    def redirection(self, i: int) -> int:
        text = self.text
        # `2>file`: the digits are the descriptor, not an argument.
        if self.in_word and "".join(self.word).isdigit():
            self.word, self.in_word = [], False
        else:
            self.end_word()
        if text[i] == "&":
            i += 1  # &>
        op = text[i]
        j = i + 1
        if op == ">" and text[j : j + 1] == ">":
            op, j = ">>", j + 1
        elif op == ">" and text[j : j + 1] == "|":
            op, j = ">|", j + 1
        elif op == "<" and text[j : j + 2] == "<<":
            op, j = "<<<", j + 2
        elif op == "<" and text[j : j + 1] == "<":
            op, j = "<<", j + 1
            if text[j : j + 1] == "-":
                op, j = "<<-", j + 1
        if text[j : j + 1] == "&" and op in (">", ">>", "<"):
            self.redirect, j = "dup", j + 1
        elif op == "<":
            self.redirect = "<"
        elif op == "<<<":
            self.redirect = "<<<"
        else:
            self.redirect = op
        # `<` and `<<<` read; only the others name a file that is written (or a heredoc delimiter).
        if self.redirect in ("<", "<<<"):
            self.redirect = "dup"
        return j

    def dollar(self, i: int) -> int:
        text, n = self.text, len(self.text)
        nxt = text[i + 1 : i + 2]
        if nxt == "(":
            if text[i + 2 : i + 3] == "(":  # arithmetic
                end = _balanced(text, i + 2)
                self.add("$(())")
                return end + 1
            end = _balanced(text, i + 2)
            self.cur.subs.append(text[i + 2 : end])
            self.add("$()")
            return end + 1
        if nxt == "{":
            end = text.find("}", i + 2)
            end = n if end < 0 else end
            inner = text[i + 2 : end]
            self.cur.subs.extend(_subs_in(inner))
            self.add("${" + inner + "}")
            return end + 1
        if nxt == "'":
            end = i + 2
            while end < n and text[end] != "'":
                end += 2 if text[end] == "\\" else 1
            self.add(_ansi_c(text[i + 2 : end]))
            self.quoted = True
            return end + 1
        if nxt == '"':
            self.quoted = True
            return self.double_quoted(i + 2)
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-]", text[i + 1 :])
        if m:
            self.add("$" + m.group(0))
            return i + 1 + len(m.group(0))
        self.add("$")
        return i + 1

    def backtick(self, i: int) -> int:
        text, n = self.text, len(self.text)
        end = i + 1
        while end < n and text[end] != "`":
            end += 2 if text[end] == "\\" else 1
        self.cur.subs.append(text[i + 1 : end])
        self.add("$()")
        return end + 1

    def double_quoted(self, i: int) -> int:
        text, n = self.text, len(self.text)
        self.in_word = True
        while i < n and text[i] != '"':
            c = text[i]
            if c == "\\" and i + 1 < n:
                nxt = text[i + 1]
                self.add(nxt if nxt in '"\\$`' else "\\" + nxt)
                if nxt == "\n":
                    self.word.pop()
                i += 2
            elif c == "$":
                i = self.dollar(i)
            elif c == "`":
                i = self.backtick(i)
            else:
                self.add(c)
                i += 1
        return i + 1

    def read_heredocs(self, i: int) -> int:
        text, n = self.text, len(self.text)
        pending, self.heredocs = self.heredocs, []
        for delimiter, quoted, strip, owner in pending:
            body: list[str] = []
            while i < n:
                end = text.find("\n", i)
                end = n if end < 0 else end
                line = text[i:end]
                i = end + 1
                if (line.lstrip("\t") if strip else line) == delimiter:
                    break
                body.append(line)
            if not quoted:
                owner.subs.extend(_subs_in("\n".join(body)))
        return i


def _balanced(text: str, i: int) -> int:
    """The index of the `)` closing a `(` whose content starts at `i`, or the end of the text."""
    depth, n = 1, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            end = text.find("'", i + 1)
            i = n if end < 0 else end + 1
            continue
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def _subs_in(text: str) -> list[str]:
    """The `$(...)` and backtick commands inside text that is data except for its substitutions (a heredoc body)."""
    found: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
        elif text.startswith("$(", i):
            end = _balanced(text, i + 2)
            found.append(text[i + 2 : end])
            i = end + 1
        elif text[i] == "`":
            end = text.find("`", i + 1)
            end = n if end < 0 else end
            found.append(text[i + 1 : end])
            i = end + 1
        else:
            i += 1
    return found


def _ansi_c(body: str) -> str:
    out: list[str] = []
    i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "a": "\a", "b": "\b", "e": "\x1b", "f": "\f", "v": "\v"}
    while i < len(body):
        c = body[i]
        if c != "\\" or i + 1 >= len(body):
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "x" and re.match(r"[0-9a-fA-F]{1,2}", body[i + 2 :]):
            digits = re.match(r"[0-9a-fA-F]{1,2}", body[i + 2 :]).group(0)  # type: ignore[union-attr]
            out.append(chr(int(digits, 16)))
            i += 2 + len(digits)
        elif nxt in "01234567":
            digits = re.match(r"[0-7]{1,3}", body[i + 1 :]).group(0)  # type: ignore[union-attr]
            out.append(chr(int(digits, 8)))
            i += 1 + len(digits)
        else:
            out.append(nxt)
            i += 2
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# Walk: commands to a verdict


def _walk(text: str, acc: _Acc, *, depth: int, cwd: str) -> str:
    """Judge every command of `text` into `acc`. Returns the working directory the text leaves behind."""
    if depth > _MAX_DEPTH:
        acc.bump("high", "nested too deeply to read")
        acc.skills_ok = False
        return cwd
    if _FORK_BOMB.search(text):
        acc.bump("catastrophic", "fork bomb")
        acc.skills_ok = False
    pipeline: list[_Cmd] = []
    for cmd in _Lexer(text).run():
        if not cmd.piped_in:
            pipeline = []
        for sub in cmd.subs:
            inner = _Acc()
            _walk(sub, inner, depth=depth + 1, cwd=cwd)
            acc.merge(inner)
            if inner.level != "low":
                acc.skills_ok = False
        words = _strip_leading(cmd.words)
        if words and words[0] == "cd":
            target = words[1] if len(words) > 1 else "~"
            cwd = _resolve(target, cwd) if "$" not in target else "/"
        verdict = _Acc()
        _judge(words, cmd, verdict, depth=depth, cwd=cwd, shell_input=bool(pipeline))
        pipeline.append(cmd)
        acc.merge(verdict)
        if verdict.level != "low":
            if verdict.skills_seen and verdict.skills_ok:
                acc.skills_seen = True
            else:
                acc.skills_ok = False
    return cwd


def _strip_leading(words: list[str]) -> list[str]:
    words = list(words)
    while words and words[0] in _LEAD_KEYWORDS:
        words.pop(0)
    if words and words[0] in _EMPTY_KEYWORDS:
        return []
    return words


def _judge(words: list[str], cmd: _Cmd, acc: _Acc, *, depth: int, cwd: str, shell_input: bool) -> None:
    for op, target in cmd.redirects:
        _redirect(op, target, acc, cwd)
    # `FOO=1 cmd`: the assignments are part of the environment of cmd.
    while words and _ASSIGN.match(words[0]):
        name = words.pop(0).split("=", 1)[0]
        if name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PATH", "BASH_ENV", "ENV", "PYTHONSTARTUP"):
            acc.bump("high", f"sets {name}")
    if not words:
        return
    fetches = any(_FETCH.search(sub) for sub in cmd.subs)
    prog_word = words[0]
    if "$" in prog_word:
        acc.bump("high", "runs a program it cannot name")
        return
    prog = posixpath.basename(prog_word) if "/" in prog_word else prog_word
    args = words[1:]

    if _skill_invocation(prog_word, prog, args, cwd):
        acc.skills_seen = True
    else:
        acc.skills_ok = False

    # Wrappers: judge what they run.
    inner = _unwrap(prog, args, acc)
    if inner is not None:
        words2 = _strip_leading(inner)
        sub = _Acc()
        _judge(words2, _Cmd(words2, [], cmd.subs, cmd.piped_in), sub, depth=depth, cwd=cwd, shell_input=shell_input)
        acc.merge(sub)
        acc.skills_ok = acc.skills_ok and sub.skills_ok
        return

    if prog not in _SHELLS and args in (["--version"], ["-V"], ["-v"], ["version"]) and not cmd.redirects:
        return
    if prog in _SHELLS or prog in ("source", ".", "eval") or _is_interpreter(prog):
        _interpreter(prog, args, cmd, acc, depth=depth, cwd=cwd, shell_input=shell_input, fetches=fetches)
        return
    _program(prog, args, acc, cwd, depth=depth)


# Wrappers return the command they run (or None when `prog` is not a wrapper).
def _unwrap(prog: str, args: list[str], acc: _Acc) -> list[str] | None:
    if prog in ("sudo", "doas", "pkexec"):
        acc.bump("high", "sudo" if prog == "sudo" else prog)
        return _skip_options(args, with_value={"-u", "-g", "-p", "-C", "-D", "-R", "-T", "-r", "-t", "-h", "-U", "--user", "--group"})
    if prog == "su":
        acc.bump("high", "su")
        if "-c" in args and args.index("-c") + 1 < len(args):
            return ["sh", "-c", args[args.index("-c") + 1]]
        return []
    if prog == "env":
        rest = _skip_options(args, with_value={"-u", "-C", "-S", "--unset", "--chdir"})
        while rest and _ASSIGN.match(rest[0]):
            rest.pop(0)
        if not rest:
            acc.bump("medium", "env dumps the environment")
        return rest
    if prog in ("nohup", "time", "setsid", "unbuffer", "builtin", "exec", "busybox", "chrt", "taskset"):
        return _skip_options(args, with_value={"-n", "-p", "-a"} if prog == "exec" else set())
    if prog in ("nice", "ionice"):
        return _skip_options(args, with_value={"-n", "-c", "-p"})
    if prog == "stdbuf":
        return _skip_options(args, with_value={"-i", "-o", "-e"})
    if prog == "timeout":
        rest = _skip_options(args, with_value={"-s", "-k", "--signal", "--kill-after"})
        return rest[1:]
    if prog == "command":
        if args and args[0] in ("-v", "-V"):
            return []
        return _skip_options(args, with_value=set())
    if prog == "xargs":
        rest = _skip_options(args, with_value={"-n", "-I", "-P", "-d", "-L", "-s", "-a", "-E", "-i", "--max-args", "--max-procs", "--delimiter"})
        return rest or ["echo"]
    return None


def _skip_options(args: list[str], *, with_value: set[str]) -> list[str]:
    out = list(args)
    while out and out[0].startswith("-") and out[0] != "-":
        flag = out.pop(0)
        if flag == "--":
            break
        if flag in with_value and out:
            out.pop(0)
    return out


def _is_interpreter(prog: str) -> bool:
    return re.sub(r"[0-9.]+$", "", prog) in _INTERPRETERS or prog in _INTERPRETERS or prog == "uv"


def _interpreter(
    prog: str, args: list[str], cmd: _Cmd, acc: _Acc, *, depth: int, cwd: str, shell_input: bool, fetches: bool
) -> None:
    is_shell = prog in _SHELLS
    if prog == "eval":
        acc.bump("high", "eval")
        sub = _Acc()
        _walk(" ".join(args), sub, depth=depth + 1, cwd=cwd)
        acc.merge(sub)
        return
    if prog == "uv" and not (args and args[0] == "run" and not any(a in ("-c", "-m") for a in args)):
        _program(prog, args, acc, cwd, depth=depth)
        return
    if is_shell and "-c" in args and args.index("-c") + 1 < len(args):
        acc.bump("medium", f"{prog} -c")
        sub = _Acc()
        _walk(args[args.index("-c") + 1], sub, depth=depth + 1, cwd=cwd)
        acc.merge(sub)
        if sub.level != "low":
            acc.skills_ok = False
    elif prog in ("source", "."):
        acc.bump("medium", "source")
    else:
        operands = [a for a in args if not a.startswith("-")]
        if shell_input and not operands:
            acc.bump("high", f"pipe into {prog}")
        else:
            acc.bump("medium", prog)
    if fetches:
        acc.bump("high", f"runs downloaded code with {prog}")


def _skill_invocation(prog_word: str, prog: str, args: list[str], cwd: str) -> bool:
    if prog_word.startswith(("/", "./")):
        return _under_skills(prog_word, cwd)
    base = re.sub(r"[0-9.]+$", "", prog)
    if prog in _SHELLS or base in _INTERPRETERS or prog == "uv":
        rest = args
        if prog == "uv":
            if not rest or rest[0] != "run":
                return False
            rest = rest[1:]
        for arg in rest:
            if arg in ("-c", "-m", "-e", "-p", "--eval", "--print"):
                return False
            if arg.startswith("-"):
                continue
            return _under_skills(arg, cwd)
    return False


def _under_skills(path: str, cwd: str) -> bool:
    if "$" in path or "*" in path:
        return False
    return _resolve(path, cwd).startswith(SKILLS_PREFIX)


# ---------------------------------------------------------------------------------------------------------------------
# Paths


def _resolve(path: str, cwd: str) -> str:
    """An absolute, normalised path for `path`, read from `cwd`. `~` and `$HOME` stay `~`."""
    path = _home(path)
    if path.startswith("~"):
        rest = path[1:].split("/", 1)
        tail = rest[1] if len(rest) > 1 else ""
        return "~" + (posixpath.normpath("/" + tail) if tail.strip("/") else "")
    if not path.startswith("/"):
        path = posixpath.join(cwd, path)
    resolved = posixpath.normpath(path)
    return "/" + resolved.lstrip("/") if resolved.startswith("//") else resolved


def _home(path: str) -> str:
    for prefix in ("${HOME}", "$HOME"):
        if path.startswith(prefix):
            return "~" + path[len(prefix) :]
    return path


def _expand_braces(word: str) -> list[str]:
    match = re.search(r"\{([^{}]*,[^{}]*)\}", word)
    if not match:
        return [word]
    out: list[str] = []
    for option in match.group(1).split(","):
        out.extend(_expand_braces(word[: match.start()] + option + word[match.end() :]))
        if len(out) > 32:
            break
    return out


def _is_root_like(path: str) -> bool:
    """A target whose recursive removal, move or re-owning wipes a machine or a home."""
    if path in ("/", "/*", "~", "~/*") or re.fullmatch(r"/[^/]*[*?\[][^/]*", path) or re.fullmatch(r"~/[^/]*[*?\[][^/]*", path):
        return True
    if path in _SYSTEM_STRICT or path in _SYSTEM_LOOSE:
        return True
    parts = path.split("/")
    if len(parts) == 3 and f"/{parts[1]}" in _SYSTEM_STRICT:
        return True
    return len(parts) == 3 and parts[2] == "*" and f"/{parts[1]}" in _SYSTEM_LOOSE


def _protected(path: str) -> bool:
    lowered = path.lower().rstrip("/")
    parts = [p for p in lowered.split("/") if p]
    if not parts:
        return False
    if any(part in _PROTECTED_DIRS for part in parts):
        return True
    base = parts[-1]
    if base in _PROTECTED_FILES or base.startswith(".env"):
        return True
    return any("/" in f and lowered.endswith("/" + f) for f in _PROTECTED_FILES)


def _device_level(path: str) -> Level | None:
    if path in _HARMLESS_DEVICES or path.startswith(("/dev/fd/", "/dev/pts/")):
        return None
    if _BLOCK_DEVICE.match(path) or path == "/proc/sysrq-trigger":
        return "catastrophic"
    if path.startswith(("/dev/tcp/", "/dev/udp/")):
        return "high"
    if path.startswith("/dev/"):
        return "high"
    return None


def _check_write_target(raw: str, acc: _Acc, cwd: str, *, verb: str) -> None:
    """A path the command writes, creates, moves or removes."""
    acc.writes = True
    if "$" in raw and not raw.startswith(("$HOME", "${HOME}")):
        return
    for variant in _expand_braces(raw):
        path = _resolve(variant, cwd)
        level = _device_level(path)
        if level is not None:
            acc.bump(level, f"{verb} {path}")
        elif _protected(path):
            acc.protected = True
            acc.bump("high", f"{verb} {path}")
        elif any(path == d or path.startswith(d + "/") for d in _SYSTEM_WRITE):
            acc.bump("high", f"{verb} {path}")


def _redirect(op: str, target: str, acc: _Acc, cwd: str) -> None:
    if target in _HARMLESS_DEVICES:
        return
    if "$" in target and not target.startswith(("$HOME", "${HOME}")):
        acc.writes = True
        acc.bump("medium", f"writes to {target}")
        return
    _check_write_target(target, acc, cwd, verb="writes to")
    acc.bump("medium", "writes a file")


# ---------------------------------------------------------------------------------------------------------------------
# Programs


def _options(args: list[str]) -> tuple[set[str], set[str], list[str]]:
    """Short flag letters, long flags and operands of `args`; everything after `--` is an operand."""
    shorts: set[str] = set()
    longs: set[str] = set()
    operands: list[str] = []
    done = False
    for arg in args:
        if done or arg == "-" or not arg.startswith("-"):
            operands.append(arg)
        elif arg == "--":
            done = True
        elif arg.startswith("--"):
            longs.add(arg.split("=", 1)[0])
        else:
            shorts.update(arg[1:])
    return shorts, longs, operands


def _program(prog: str, args: list[str], acc: _Acc, cwd: str, *, depth: int) -> None:
    for arg in args:
        if arg.startswith(("/dev/tcp/", "/dev/udp/")):
            acc.bump("high", "network through /dev/tcp")
    shorts, longs, operands = _options(args)

    if prog == "rm" or prog in ("unlink", "shred", "rmdir"):
        _rm(prog, args, shorts, longs, operands, acc, cwd)
    elif prog.startswith("mkfs") or prog in _DISK_FORMAT:
        acc.bump("catastrophic", f"{prog} formats a disk")
    elif prog in _POWER:
        acc.bump("catastrophic", prog)
    elif prog in ("init", "telinit"):
        acc.bump("catastrophic" if operands[:1] in (["0"], ["6"]) else "high", f"{prog} {' '.join(operands[:1])}".strip())
    elif prog == "systemctl":
        _systemctl(operands, acc)
    elif prog == "service":
        acc.bump("medium" if operands[1:2] == ["status"] else "high", "service")
    elif prog == "dd":
        _dd(args, acc, cwd)
    elif prog in ("chmod", "chown", "chgrp"):
        _chmod(prog, shorts, longs, operands, acc, cwd)
    elif prog == "mv":
        for operand in operands:
            for variant in _expand_braces(operand):
                if _is_root_like(_resolve(variant, cwd)):
                    acc.bump("catastrophic", f"mv {operand}")
        for operand in operands:
            _check_write_target(operand, acc, cwd, verb="moves")
        acc.bump("medium", "mv")
    elif prog in ("cp", "install", "ln"):
        dest = operands[-1:] if "t" not in shorts and "-t" not in args else operands
        for operand in dest:
            _check_write_target(operand, acc, cwd, verb="writes")
        acc.writes = True
        acc.bump("medium", prog)
    elif prog in ("touch", "mkdir", "tee", "truncate", "mkfifo", "mknod", "patch"):
        for operand in operands:
            _check_write_target(operand, acc, cwd, verb="writes")
        acc.writes = True
        acc.bump("medium", prog)
    elif prog == "sed":
        in_place = "i" in shorts or "--in-place" in longs or any(a.startswith("-i") for a in args)
        if in_place:
            targets = operands if ("e" in shorts or "f" in shorts) else operands[1:]
            for operand in targets:
                _check_write_target(operand, acc, cwd, verb="edits")
            acc.writes = True
        acc.bump("medium", "sed -i" if in_place else "sed")
    elif prog == "find":
        _find(args, acc, cwd, depth=depth)
    elif prog == "git":
        _git(args, acc, cwd)
    elif prog in ("curl", "wget"):
        _fetch(prog, args, shorts, longs, acc)
    elif prog == "rsync":
        acc.writes = True
        remote = any(re.match(r"^([^/\s]+@)?[^/\s:]+:", a) or a.startswith("rsync://") for a in operands)
        acc.bump("high" if remote else "medium", "rsync to a remote host" if remote else "rsync")
        if not remote and operands:
            _check_write_target(operands[-1], acc, cwd, verb="writes")
    elif prog in ("kill", "pkill", "killall"):
        _kill(prog, args, acc)
    elif prog in ("npm", "pnpm", "yarn", "bun", "npx"):
        _node_packages(prog, args, longs, shorts, operands, acc)
    elif prog in ("pip", "pip3", "pipx") or (prog.startswith("pip") and prog[3:].replace(".", "").isdigit()):
        _pip(prog, args, longs, operands, acc)
    elif prog in _PKG_SYSTEM:
        _system_packages(prog, args, operands, acc)
    elif prog in ("gem", "cargo") and operands[:1] == ["install"]:
        acc.bump("high", f"{prog} install (global)")
    elif prog == "crontab":
        acc.bump("medium" if "l" in shorts else "high", "crontab")
    elif prog in ("docker", "podman"):
        _docker(prog, args, operands, acc)
    elif prog in ("env", "printenv", "set"):
        acc.bump("medium", f"{prog} dumps the environment") if prog != "set" else None
    elif prog in ("tar",):
        flags = "".join(shorts) + ("".join(a[1:] for a in args[:1] if not a.startswith("-")))
        if "t" in flags and "x" not in flags and "c" not in flags:
            return
        acc.writes = True
        acc.bump("medium", "tar")
    elif prog == "sort" and ("o" in shorts or "--output" in longs):
        acc.writes = True
        acc.bump("medium", "sort -o")
    elif prog == "date" and ("s" in shorts or "--set" in longs):
        acc.bump("high", "date -s")
    elif prog in ("hostname",) and operands:
        acc.bump("high", "hostname")
    elif prog in ("python", "node") or _is_interpreter(prog):
        acc.bump("medium", prog)
    elif args in (["--version"], ["-V"], ["-v"], ["--help"], ["-h"], ["version"]) or prog in _LOW_PROGRAMS:
        return
    else:
        if prog in _HIGH_PROGRAMS:
            acc.bump("high", _HIGH_PROGRAMS[prog])
        else:
            acc.bump("medium", prog)
        if prog in _WRITE_PROGRAMS:
            acc.writes = True
    if prog in _HIGH_PROGRAMS and prog not in ("killall", "pkill"):
        acc.bump("high", _HIGH_PROGRAMS[prog])
    if prog in _WRITE_PROGRAMS:
        acc.writes = True


def _rm(prog: str, args: list[str], shorts: set[str], longs: set[str], operands: list[str], acc: _Acc, cwd: str) -> None:
    acc.writes = True
    recursive = bool(shorts & {"r", "R"}) or "--recursive" in longs
    force = "f" in shorts or "--force" in longs
    no_preserve = "--no-preserve-root" in longs
    label = "rm -" + ("r" if recursive else "") + ("f" if force else "") if (recursive or force) else prog
    for operand in operands:
        if "$" in operand and not operand.startswith(("$HOME", "${HOME}")):
            if recursive:
                acc.bump("high", f"{label} {operand}")
            continue
        for variant in _expand_braces(operand):
            path = _resolve(variant, cwd)
            if (recursive or no_preserve) and _is_root_like(path):
                acc.bump("catastrophic", f"{label} {operand}")
            elif path in (WORKSPACE, WORKSPACE + "/*"):
                acc.bump("high", f"{prog} wipes the workspace")
            else:
                _check_write_target(variant, acc, cwd, verb="removes")
    if recursive and force:
        acc.bump("high", "rm -rf")
    else:
        acc.bump("medium", prog)


def _dd(args: list[str], acc: _Acc, cwd: str) -> None:
    acc.writes = True
    out = next((a[3:] for a in args if a.startswith("of=")), "")
    level = _device_level(out) if out else None
    if level == "catastrophic":
        acc.bump("catastrophic", f"dd of={out}")
    elif out in _HARMLESS_DEVICES:
        acc.bump("medium", "dd")
    else:
        if out:
            _check_write_target(out, acc, cwd, verb="dd to")
        acc.bump("high", "dd")


def _chmod(prog: str, shorts: set[str], longs: set[str], operands: list[str], acc: _Acc, cwd: str) -> None:
    acc.writes = True
    recursive = bool(shorts & {"R"}) or "--recursive" in longs
    # `chmod MODE FILE...`, `chown OWNER FILE...`; `--reference=X` names no mode.
    targets = operands[1:] if operands else []
    mode = operands[0] if operands else ""
    for target in targets:
        for variant in _expand_braces(target):
            path = _resolve(variant, cwd)
            if recursive and _is_root_like(path):
                acc.bump("catastrophic", f"{prog} -R {target}")
        _check_write_target(target, acc, cwd, verb="changes")
    if recursive:
        acc.bump("high", f"{prog} -R")
    if prog == "chmod" and re.fullmatch(r"0?777|[ugoa]*[+=]rwx|a\+w|o\+w", mode):
        acc.bump("high", "chmod 777")
    acc.bump("medium", prog)


def _systemctl(operands: list[str], acc: _Acc) -> None:
    verb = operands[0] if operands else ""
    if verb in ("poweroff", "reboot", "halt", "kexec", "rescue", "emergency"):
        acc.bump("catastrophic", f"systemctl {verb}")
    elif verb in ("status", "show", "is-active", "is-enabled", "is-failed", "list-units", "list-unit-files", "list-timers", "cat", "list-dependencies", ""):
        return
    else:
        acc.bump("high", f"systemctl {verb}")


def _kill(prog: str, args: list[str], acc: _Acc) -> None:
    signal_flags = [a for a in args if re.fullmatch(r"-(9|KILL|SIGKILL)", a)]
    spaced = any(args[i] == "-s" and args[i + 1] in ("9", "KILL", "SIGKILL") for i in range(len(args) - 1))
    pids = [a for a in args if not a.startswith("-") or a == "-1"]
    if prog == "kill" and ("-1" in args[1:] or "1" in pids):
        acc.bump("catastrophic", "kill -1" if "-1" in args[1:] else "kill 1")
        return
    if prog in ("pkill", "killall"):
        acc.bump("high", prog)
    elif signal_flags or spaced:
        acc.bump("high", "kill -9")
    else:
        acc.bump("medium", "kill")


def _find(args: list[str], acc: _Acc, cwd: str, *, depth: int) -> None:
    starts: list[str] = []
    for arg in args:
        if arg.startswith(("-", "(", "!")):
            break
        starts.append(arg)
    destructive = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "-delete":
            acc.writes = True
            acc.bump("high", "find -delete")
            destructive = True
        elif arg in ("-exec", "-execdir", "-ok", "-okdir"):
            j = i + 1
            while j < len(args) and args[j] not in (";", "+", "\\;"):
                j += 1
            inner = _Acc()
            _walk(" ".join(args[i + 1 : j]), inner, depth=depth + 1, cwd=cwd)
            acc.merge(inner)
            acc.bump("medium", "find -exec")
            destructive = destructive or _RANK[inner.level] >= _RANK["high"]
            i = j
        elif arg.startswith(("-fprint", "-fls")):
            acc.writes = True
            acc.bump("medium", "find writes a file")
        i += 1
    if destructive:
        for start in starts:
            if "$" not in start and _is_root_like(_resolve(start, cwd)):
                acc.bump("catastrophic", f"find {start} (destructive)")


def _git(args: list[str], acc: _Acc, cwd: str) -> None:
    rest = list(args)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") and rest:
            rest.pop(0)
    sub = rest[0] if rest else ""
    shorts, longs, operands = _options(rest[1:])
    flags = shorts | longs
    if sub == "push" and (shorts & {"f"} or longs & {"--force", "--force-with-lease", "--mirror", "--delete"} or any(o.startswith("+") for o in operands)):
        acc.bump("high", "git push --force")
    elif sub == "reset" and "--hard" in longs:
        acc.writes = True
        acc.bump("high", "git reset --hard")
    elif sub == "clean" and ("f" in shorts or "--force" in longs):
        acc.writes = True
        acc.bump("high", "git clean -f")
    elif sub == "config":
        global_scope = bool(longs & {"--global", "--system"})
        reading = bool(longs & {"--get", "--get-all", "--get-regexp", "--list"} or shorts & {"l"})
        if not reading:
            acc.writes = True
            if global_scope:
                acc.protected = True
                acc.bump("high", "git config --global")
            else:
                acc.bump("medium", "git config")
    elif sub in _GIT_LOW and not (sub == "grep" and ("O" in shorts or longs & {"--open-files-in-pager", "--output"})) or sub == "branch" and not operands and not (shorts & {"d", "D", "m", "M", "c", "C", "u"}) and not (longs & {"--delete", "--move", "--copy", "--set-upstream-to"}) or sub == "tag" and (not operands or shorts & {"l"} or "--list" in longs) and not (shorts & {"d", "a", "s", "f"}) or sub == "remote" and (not operands or operands == ["show"] or operands[:1] == ["-v"]) and not flags - {"v"} or sub in ("stash", "reflog") and (not operands or operands[0] in ("list", "show")) or sub == "":
        return
    if sub not in _NO_WRITE_GIT and not (sub in ("branch", "tag", "remote", "stash", "reflog") and acc.level == "low"):
        acc.writes = True
    acc.bump("medium", f"git {sub}")


def _fetch(prog: str, args: list[str], shorts: set[str], longs: set[str], acc: _Acc) -> None:
    upload = False
    if prog == "curl":
        upload = bool(shorts & {"d", "F", "T"} or longs & {"--data", "--data-binary", "--data-raw", "--data-urlencode", "--data-ascii", "--form", "--form-string", "--upload-file", "--json"})
        for i, arg in enumerate(args):
            if arg in ("-X", "--request") and i + 1 < len(args) and args[i + 1].upper() not in ("GET", "HEAD"):
                upload = True
            if arg.upper().startswith("-X") and len(arg) > 2 and arg[2:].upper() not in ("GET", "HEAD"):
                upload = True
        if shorts & {"o", "O"} or longs & {"--output", "--remote-name"}:
            acc.writes = True
    else:
        upload = any(a.startswith(("--post-data", "--post-file", "--body-data", "--body-file")) for a in args)
        upload = upload or any(a.upper() in ("--METHOD=POST", "--METHOD=PUT", "--METHOD=PATCH", "--METHOD=DELETE") for a in args)
        acc.writes = True
    if upload:
        acc.bump("high", f"{prog} uploads data")
    else:
        acc.bump("medium", prog)


def _node_packages(prog: str, args: list[str], longs: set[str], shorts: set[str], operands: list[str], acc: _Acc) -> None:
    verb = operands[0] if operands else ""
    if verb in ("ls", "list", "view", "outdated", "info", "why", "audit", "root", "bin", "") and not shorts - {"v"}:
        return
    if "g" in shorts or "--global" in longs or verb == "global" or "--location=global" in args:
        acc.bump("high", f"{prog} global install")
    else:
        acc.writes = verb in ("install", "i", "add", "ci", "update", "remove", "uninstall", "rm", "link", "init")
        acc.bump("medium", prog)


def _pip(prog: str, args: list[str], longs: set[str], operands: list[str], acc: _Acc) -> None:
    verb = operands[0] if operands else ""
    if verb in ("list", "show", "freeze", "check", "search", "") or args in (["--version"],):
        return
    if longs & {"--user", "--break-system-packages", "--root"} or "--target=/" in args:
        acc.bump("high", f"{prog} install outside the workspace")
    else:
        acc.writes = True
        acc.bump("medium", prog)


def _system_packages(prog: str, args: list[str], operands: list[str], acc: _Acc) -> None:
    verb = operands[0] if operands else ""
    if verb in _PKG_READ or any(a in _PKG_READ for a in args[:1]):
        return
    acc.bump("high", f"{prog} changes system packages")


def _docker(prog: str, args: list[str], operands: list[str], acc: _Acc) -> None:
    verb = operands[0] if operands else ""
    text = " ".join(args)
    if "--privileged" in args or "--pid=host" in args or "--net=host" in args or "docker.sock" in text or re.search(r"(^|\s)-v\s*/:|--volume[= ]/:|(^|\s)-v\s*/(etc|root|var|usr):", text):
        acc.bump("high", f"{prog} with host access")
    elif verb in ("ps", "images", "inspect", "logs", "version", "info", "images", "top", "stats", ""):
        return
    else:
        acc.bump("medium", prog)
