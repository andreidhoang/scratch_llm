#!/usr/bin/env bash
# noise_guard.sh — refuse to commit or push noise. Structure, not prose.
#
#   Installed as .git/hooks/pre-commit and .git/hooks/pre-push.
#   Re-arm after a fresh clone (hooks are not tracked by git):
#       bash <path-to-this-file> --install
#   Deliberate override, and say why in the commit message:
#       NOISE_GUARD=off git commit ...
#
# Why this exists: these repos are PUBLIC. A single `git add .` in a root that holds
# a study shelf puts third-party books and 20+ MB of binaries into a public history
# permanently — git keeps blobs after deletion. .gitignore alone does not stop
# `git add -f`, a file already tracked, or a push from another machine.
set -uo pipefail

[ "${NOISE_GUARD:-on}" = off ] && exit 0
MODE="${1:-precommit}"
[ "$MODE" = "--install" ] && MODE=install
MAX_BYTES="${NOISE_GUARD_MAX_BYTES:-2000000}"   # 2 MB

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [ "$MODE" = install ]; then
  for h in pre-commit pre-push; do
    m=precommit; [ "$h" = pre-push ] && m=prepush
    hp="$ROOT/.git/hooks/$h"
    # NEVER clobber an existing hook. A redirect onto a SYMLINK writes through to its
    # target, which would silently destroy a shared hook script. (This happened once,
    # 2026-09-14, to scratch_llm's green-ci-gate.sh. Hence the check.)
    if [ -e "$hp" ] || [ -L "$hp" ]; then
      tgt="$hp"; [ -L "$hp" ] && tgt="$(readlink -f "$hp" 2>/dev/null || echo "$hp")"
      if grep -q noise_guard "$tgt" 2>/dev/null; then
        echo "ok    $hp (guard already chained)"
      else
        echo "SKIP  $hp already exists -> $tgt"
        echo "      Chain it by hand, near the top of that file:"
        echo "        R=\"\$(git rev-parse --show-toplevel)\""
        echo "        case \"\$(basename \"\$0\")\" in pre-push) NG_MODE=prepush;; *) NG_MODE=precommit;; esac"
        echo "        [ -x \"\$R/${SELF#$ROOT/}\" ] && { \"\$R/${SELF#$ROOT/}\" \"\$NG_MODE\" || exit 1; }"
      fi
      continue
    fi
    # RELOCATABLE (fixed 2026-09-16): the hook resolves its own repo at RUN time. Baking the
    # absolute $SELF here is what broke every commit in ladders and reasoningLLM — the path was
    # a Cowork session mount that no longer existed, so the hook exited 126 and git refused.
    rel="${SELF#$ROOT/}"
    cat > "$hp" <<HOOK
#!/usr/bin/env bash
R="\$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
G="\$R/$rel"
[ -x "\$G" ] || { echo "noise_guard: \$G missing - run 'make guard' to re-arm" >&2; exit 0; }
exec "\$G" $m
HOOK
    chmod +x "$hp"
    echo "armed $hp"
  done
  exit 0
fi

DENY_PATH='(^|/)(mastery|\.venv|venv|node_modules|__pycache__|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.pyright|htmlcov|wandb|checkpoints|\.huggingface|\.cache|eval_results|_to_delete|\.reps|\.zcode|\.ipynb_checkpoints)(/|$)'
DENY_EXT='\.(pdf|epub|mobi|djvu|ckpt|pt|pth|safetensors|gguf|onnx|so|dylib|dll|pyd|whl|zip|tar|gz|tgz|bz2|xz|7z|rar|mp4|mov|avi|mkv|ncu-rep|nsys-rep|qdrep|sqlite|pyc|pyo|cubin|fatbin)$'
DENY_NAME='(^|/)(\.DS_Store|Thumbs\.db|\.env|.*\.key|.*\.pem|id_rsa.*)$'

list_files() {
  if [ "$MODE" = prepush ]; then
    while read -r _l lsha _r rsha; do
      [ "$lsha" = "0000000000000000000000000000000000000000" ] && continue
      if [ "$rsha" = "0000000000000000000000000000000000000000" ]; then
        git rev-list "$lsha" --not --remotes 2>/dev/null | while read -r c; do git show --pretty= --name-only "$c"; done
      else
        git diff --name-only "$rsha" "$lsha" 2>/dev/null
      fi
    done | sort -u
  else
    git diff --cached --name-only --diff-filter=ACM
  fi
}

bad=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  why=""
  echo "$f" | grep -Eq "$DENY_PATH" && why="denied path"
  [ -z "$why" ] && echo "$f" | grep -Eiq "$DENY_EXT" && why="denied extension"
  [ -z "$why" ] && echo "$f" | grep -Eq "$DENY_NAME" && why="denied filename"
  if [ -z "$why" ]; then
    sz="$(git cat-file -s ":$f" 2>/dev/null || { [ -f "$f" ] && wc -c <"$f"; } || echo 0)"
    [ "${sz:-0}" -gt "$MAX_BYTES" ] && why="$((sz/1024)) KB > $((MAX_BYTES/1024)) KB"
  fi
  if [ -n "$why" ]; then
    [ "$bad" -eq 0 ] && { echo; echo "  NOISE GUARD — $MODE blocked. These are public repos; git keeps blobs forever."; echo; }
    printf '    %-60s  %s\n' "$f" "$why"
    bad=$((bad+1))
  fi
done < <(list_files)

if [ "$bad" -gt 0 ]; then
  cat <<'MSG'

  Fix, in order of preference:
    1. It is noise      -> add the path to .gitignore, then:  git rm --cached <path>
    2. It is a big file -> keep it out; commit the derived numbers, not the blob
    3. It is genuinely  -> raise the cap for this one commit:
       source + needed     NOISE_GUARD_MAX_BYTES=<bytes> git commit ...
    4. You are certain  -> NOISE_GUARD=off git commit ...   (and say why in the message)

  Never use `git add .` in these repos. Stage explicit paths: git add <file> <file>
MSG
  exit 1
fi
exit 0
