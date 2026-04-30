#!/usr/bin/env bash
# Reverse the SAPIEN _vulkan_tricks.py patch (restore pre-fix behavior).
# Used ONLY for the S1 verification experiment in env RoboFactory_prefix.
# DO NOT run on the main RoboFactory env.
set -euo pipefail

CURRENT_ENV="${CONDA_DEFAULT_ENV:-}"
if [ "$CURRENT_ENV" != "RoboFactory_prefix" ]; then
    echo "ERROR: refusing to run outside conda env RoboFactory_prefix (current: $CURRENT_ENV)" >&2
    exit 1
fi

SAPIEN_FILE="$(python -c 'import sapien, os; print(os.path.join(os.path.dirname(sapien.__file__), "_vulkan_tricks.py"))')"
echo "Reverse-patching: $SAPIEN_FILE"

if grep -q '"/usr/share/vulkan/icd.d/nvidia_icd.json",' "$SAPIEN_FILE" && \
   grep -q '"/etc/vulkan/icd.d/nvidia_icd.json",' "$SAPIEN_FILE"; then
    :
else
    echo "Patched form not detected; nothing to revert (or already reverted)."
    exit 0
fi

python - "$SAPIEN_FILE" <<'PY'
import sys

p = sys.argv[1]
src = open(p).read()

new = ('    # Check both canonical system paths. IRIS nodes (Ubuntu 20.04 + manual NVIDIA\n'
       '    # driver install) put the ICD at /etc/vulkan/icd.d/ rather than /usr/share/.\n'
       '    for candidate in (\n'
       '        "/usr/share/vulkan/icd.d/nvidia_icd.json",\n'
       '        "/etc/vulkan/icd.d/nvidia_icd.json",\n'
       '    ):\n'
       '        if os.path.isfile(candidate):\n'
       '            os.environ["VK_ICD_FILENAMES"] = candidate\n'
       '            return')

old = ('    if os.path.exists("/usr/share/vulkan/icd.d") and os.path.isfile(\n'
       '        "/usr/share/vulkan/icd.d/nvidia_icd.json"\n'
       '    ):\n'
       '        return')

if new not in src:
    print("ERROR: patched anchor not found.", file=sys.stderr)
    sys.exit(2)

open(p, "w").write(src.replace(new, old))
print("Reverse-patch applied. Pre-fix behavior restored.")
PY

# Remove the activate.d hook so VK_ICD_FILENAMES does NOT get exported.
ENV_DIR="$(python -c 'import sys; print(sys.prefix)')"
HOOK="$ENV_DIR/etc/conda/activate.d/vulkan_icd.sh"
DEHOOK="$ENV_DIR/etc/conda/deactivate.d/vulkan_icd.sh"
rm -f "$HOOK" "$DEHOOK"
echo "Removed activate/deactivate vulkan_icd hooks (if any)."

python -c "import sapien; print('sapien import OK in pre-fix mode')"
