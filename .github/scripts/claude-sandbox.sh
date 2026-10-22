#!/usr/bin/env bash
# Prepare Bubblewrap for Claude Code's subprocess isolation
# (CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1) on a GitHub-hosted Ubuntu runner, and
# prove that a command runs inside it before Claude is invoked.
#
# Claude Code only checks that a bwrap binary exists; an unusable one fails
# every Bash call at run time instead. On Ubuntu 24.04, AppArmor denies
# unprivileged user namespaces their capabilities unless the executable has a
# profile granting `userns`, so this script installs such a profile for bwrap
# only when the first probe fails under that restriction.
#
# Run from the trusted controller checkout. Writes result=ok|failed to
# $GITHUB_OUTPUT. See docs/agent-workflow.md ("Subprocess sandbox").
#
# CLAUDE_SANDBOX_RESTRICT_FILE and CLAUDE_SANDBOX_APPARMOR_DIR exist only for
# local tests; the workflow does not set them.
set -euo pipefail

restrict_file=${CLAUDE_SANDBOX_RESTRICT_FILE:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}
apparmor_dir=${CLAUDE_SANDBOX_APPARMOR_DIR:-/etc/apparmor.d}
profile_name=claude-code-bwrap

work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/claude-sandbox.XXXXXX")
trap 'rm -rf "$work"' EXIT

result() {
  if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "result=$1" >> "$GITHUB_OUTPUT"; fi
}

fail() {
  echo "::error title=Claude sandbox preflight failed::$1"
  result failed
  exit 1
}

# First non-empty line of the last probe's output, for one-line annotations.
probe_error() {
  grep -m1 -v '^[[:space:]]*$' "$work/out" 2>/dev/null || echo "no output"
}

show_probe() {
  echo "::group::$1"
  cat "$work/out" 2>/dev/null || true
  echo "::endgroup::"
}

# Run a shell inside the same kinds of namespaces and mounts Claude Code's
# sandbox uses, and require its write through a bind mount to be visible
# outside.
probe() {
  rm -rf "$work/bind" "$work/out"
  mkdir "$work/bind"
  # shellcheck disable=SC2016 # $1 is expanded by the sandboxed shell
  "$bwrap" --die-with-parent --new-session \
    --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-net \
    --ro-bind / / --dev /dev --proc /proc \
    --bind "$work/bind" "$work/bind" \
    -- /bin/sh -c 'echo sandbox-ok > "$1/marker"' sh "$work/bind" \
    > "$work/out" 2>&1 || return 1
  [ "$(cat "$work/bind/marker" 2>/dev/null)" = sandbox-ok ]
}

if ! command -v bwrap > /dev/null || ! command -v socat > /dev/null; then
  { sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends bubblewrap socat; } ||
    fail "could not install bubblewrap and socat"
fi
bwrap=$(readlink -f "$(command -v bwrap)")
case $bwrap in
  *[!A-Za-z0-9/._+-]*) fail "unexpected bwrap path: $bwrap" ;;
esac
restrict=$(cat "$restrict_file" 2> /dev/null || echo unavailable)
echo "bwrap: $bwrap ($("$bwrap" --version 2>&1 || true))"
echo "kernel: $(uname -r)"
echo "kernel.apparmor_restrict_unprivileged_userns: $restrict"

if probe; then
  echo "Sandbox probe passed; no system change needed."
  result ok
  exit 0
fi
show_probe "Initial sandbox probe output"
echo "::notice title=Claude sandbox::Initial bwrap probe failed: $(probe_error)"

if [ "$restrict" != 1 ]; then
  fail "bwrap cannot run a command, and the AppArmor user-namespace restriction is not active ($restrict): $(probe_error)"
fi

cat > "$work/profile" << EOF
# Installed by .github/scripts/claude-sandbox.sh: lets Bubblewrap create user
# namespaces under kernel.apparmor_restrict_unprivileged_userns=1.
abi <abi/4.0>,
include <tunables/global>

profile $profile_name $bwrap flags=(unconfined) {
  userns,
}
EOF
sudo install -m 0644 "$work/profile" "$apparmor_dir/$profile_name" ||
  fail "could not install AppArmor profile $apparmor_dir/$profile_name"
sudo apparmor_parser -r "$apparmor_dir/$profile_name" ||
  fail "could not load AppArmor profile $apparmor_dir/$profile_name"
echo "Loaded AppArmor profile $profile_name granting userns to $bwrap."

if probe; then
  echo "Sandbox probe passed with the AppArmor profile."
  result ok
  exit 0
fi
show_probe "Sandbox probe output with the AppArmor profile"
fail "bwrap cannot run a command even with the AppArmor profile: $(probe_error)"
