#!/usr/bin/env bash
# Shared tokenizer guard — source this, don't execute it.
#
# WHY THIS EXISTS. The ClimbMix BPE tokenizer that produced the s1-s7 sweep was lost with its
# pod. It was "deterministically rebuildable" in theory; in practice a library/corpus drift
# shifted bytes/token 4.08 -> 1.93 and silently moved every bpb measurement by ~2.4x, which is
# what invalidated that whole sweep (docs/S35_DATA_PROVENANCE.md). assets/tokenizers/climbmix/
# holds the canonical rebuilt copy and PROVENANCE.md pins its md5 — but until 2026-08-31 that
# pin was advisory: run_s35_phase1.sh REBUILT the tokenizer instead of copying it, reproducing
# the exact failure mode the provenance doc was written to prevent. This file makes the pin
# executable. A run that cannot prove it is using the pinned bytes must not start.
CANON_TOKENIZER="assets/tokenizers/climbmix/tokenizer.json"
CANON_TOKENIZER_MD5="4fc61379fc4bbaee73842a4aa8752a02"

md5of() { # portable: md5sum (Linux pods) / md5 -q (macOS)
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'; else md5 -q "$1"; fi
}

# require_pinned_tokenizer <dest> — put the canonical tokenizer at <dest>, or die.
# Never rebuilds. An existing file whose md5 differs is a HARD FAIL, not a skip: silently
# accepting an unverified tokenizer is precisely how the old sweep was corrupted.
require_pinned_tokenizer() {
  local dest=$1 got
  if [ ! -f "$CANON_TOKENIZER" ]; then
    echo "FATAL: canonical tokenizer missing at $CANON_TOKENIZER (it is tracked in git — bad checkout?)" >&2
    return 1
  fi
  got=$(md5of "$CANON_TOKENIZER")
  if [ "$got" != "$CANON_TOKENIZER_MD5" ]; then
    echo "FATAL: $CANON_TOKENIZER md5 $got != pinned $CANON_TOKENIZER_MD5" >&2
    return 1
  fi
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    got=$(md5of "$dest")
    if [ "$got" != "$CANON_TOKENIZER_MD5" ]; then
      echo "FATAL: $dest exists with md5 $got, pinned is $CANON_TOKENIZER_MD5." >&2
      echo "       Refusing to run on an unverified tokenizer — this is the s1-s7 corruption mode." >&2
      echo "       Delete it and re-run to install the canonical copy." >&2
      return 1
    fi
    echo "tokenizer OK: $dest verified against pin ($CANON_TOKENIZER_MD5)"
  else
    cp "$CANON_TOKENIZER" "$dest"
    echo "tokenizer OK: copied $CANON_TOKENIZER -> $dest (md5 $CANON_TOKENIZER_MD5 verified)"
  fi
}

# report_staged_tokenizer <path> — advisory. Staging re-serialises through the tokenizer
# library, so these bytes need not equal the pin; a difference is reported, not fatal.
report_staged_tokenizer() {
  local p=$1 got
  [ -f "$p" ] || { echo "FATAL: staged tokenizer not found at $p" >&2; return 1; }
  got=$(md5of "$p")
  if [ "$got" = "$CANON_TOKENIZER_MD5" ]; then echo "staged tokenizer $p: byte-identical to pin"
  else echo "staged tokenizer $p: md5 $got (re-serialised from the pinned source; not byte-identical — expected)"; fi
}
