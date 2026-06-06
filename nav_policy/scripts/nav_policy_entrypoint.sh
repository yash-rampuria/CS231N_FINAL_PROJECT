#!/usr/bin/env bash
# nav_policy container entrypoint.
#
# Responsibilities (in order):
#   1. Create a non-root user matching the host UID/GID so bind-mounted files
#      stay host-owned.
#   2. Install figs and nav_policy as editable packages (idempotent and silent
#      after the first run).
#   3. Exec the user's command (CMD or `docker compose run` override) as that
#      non-root user.

set -e

HOST_UID="$(stat -c '%u' /workspace/nav_policy)"
HOST_GID="$(stat -c '%g' /workspace/nav_policy)"

if ! getent passwd "$HOST_UID" >/dev/null 2>&1; then
    groupadd -g "$HOST_GID" nav 2>/dev/null || true
    useradd -u "$HOST_UID" -g "$HOST_GID" -m -s /bin/bash nav 2>/dev/null || true
fi
USERNAME="$(getent passwd "$HOST_UID" | cut -d: -f1)"

# Repair root-owned egg-info dirs from earlier runs.
find /workspace -maxdepth 3 -name '*.egg-info' -not -user "$HOST_UID" \
    -exec chown -R "$HOST_UID:$HOST_GID" {} + 2>/dev/null || true

# Repair root-owned data artifacts (checkpoints, rl videos, eval outputs).
find /workspace/nav_policy/data -maxdepth 3 ! -user "$HOST_UID" \
    -exec chown -R "$HOST_UID:$HOST_GID" {} + 2>/dev/null || true

# Windows bind mounts inject desktop.ini into scene/checkpoint dirs; nerfstudio's
# eval_setup parses every filename in nerfstudio_models/ as step-XXXXX.ckpt.
find /workspace/FiGS-Standalone/3dgs/workspace -name 'desktop.ini' -delete 2>/dev/null || true

# Patch nerfstudio pil_to_numpy for Pillow >= 10.0.0 compatibility.
# Pillow 10 changed ImagingCore.setimage() to require explicit extents.
# The fix is to pass (0, 0, width, height) as the second argument.
# This sed is idempotent: it only rewrites lines that still have the old form.
NS_DATA_UTILS="/usr/local/lib/python3.10/dist-packages/nerfstudio/data/utils/data_utils.py"
if [ -f "$NS_DATA_UTILS" ] && grep -q 'e\.setimage(im\.im)' "$NS_DATA_UTILS"; then
    sed -i 's/e\.setimage(im\.im)/e.setimage(im.im, (0, 0) + im.size)/' "$NS_DATA_UTILS"
fi

INSTALL_LOG=/tmp/nav_policy_install.log
if ! python -c 'import figs' >/dev/null 2>&1; then
    runuser -u "$USERNAME" -- python -m pip install -e /workspace/FiGS-Standalone --no-deps -q \
        >"$INSTALL_LOG" 2>&1 || {
            echo "FATAL: failed to install figs" >&2
            tail -n 30 "$INSTALL_LOG" >&2
            exit 1
        }
fi
if ! python -c 'import gemsplat' >/dev/null 2>&1; then
    runuser -u "$USERNAME" -- python -m pip install -e /workspace/FiGS-Standalone/gemsplat --no-deps -q \
        >>"$INSTALL_LOG" 2>&1 || {
            echo "FATAL: failed to install gemsplat" >&2
            tail -n 30 "$INSTALL_LOG" >&2
            exit 1
        }
fi
if ! python -c 'import nav_policy' >/dev/null 2>&1; then
    runuser -u "$USERNAME" -- python -m pip install -e /workspace/nav_policy --no-deps -q \
        >>"$INSTALL_LOG" 2>&1 || {
            echo "FATAL: failed to install nav_policy" >&2
            tail -n 30 "$INSTALL_LOG" >&2
            exit 1
        }
fi
if ! python -c 'import pytorch_grad_cam' >/dev/null 2>&1; then
    runuser -u "$USERNAME" -- python -m pip install grad-cam -q \
        >>"$INSTALL_LOG" 2>&1 || {
            echo "FATAL: failed to install grad-cam" >&2
            tail -n 30 "$INSTALL_LOG" >&2
            exit 1
        }
fi
# When no command was supplied, drop into an interactive login shell.
if [ "$#" -eq 0 ]; then
    set -- bash -l
fi

exec runuser -u "$USERNAME" -- "$@"
