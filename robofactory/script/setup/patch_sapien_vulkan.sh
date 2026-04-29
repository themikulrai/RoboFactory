#!/usr/bin/env bash
# Re-apply SAPIEN _vulkan_tricks.py patch idempotently.
# Run after `pip install`, env recreate, or environment migration.
set -euo pipefail

SAPIEN_FILE="$(python -c 'import sapien, os; print(os.path.join(os.path.dirname(sapien.__file__), "_vulkan_tricks.py"))')"
echo "Patching: $SAPIEN_FILE"

if grep -q "/etc/vulkan/icd.d/nvidia_icd.json" "$SAPIEN_FILE"; then
    echo "Already patched. Nothing to do."
    exit 0
fi

python - "$SAPIEN_FILE" <<'PY'
import sys

sapien_file = sys.argv[1]
src = open(sapien_file).read()

old = ('    if os.path.exists("/usr/share/vulkan/icd.d") and os.path.isfile(\n'
       '        "/usr/share/vulkan/icd.d/nvidia_icd.json"\n'
       '    ):\n'
       '        return')

new = ('    # Check both canonical system paths. IRIS nodes (Ubuntu 20.04 + manual NVIDIA\n'
       '    # driver install) put the ICD at /etc/vulkan/icd.d/ rather than /usr/share/.\n'
       '    for candidate in (\n'
       '        "/usr/share/vulkan/icd.d/nvidia_icd.json",\n'
       '        "/etc/vulkan/icd.d/nvidia_icd.json",\n'
       '    ):\n'
       '        if os.path.isfile(candidate):\n'
       '            os.environ["VK_ICD_FILENAMES"] = candidate\n'
       '            return')

if old not in src:
    print("ERROR: anchor not found; SAPIEN version may have changed.", file=sys.stderr)
    sys.exit(2)

open(sapien_file, "w").write(src.replace(old, new))
print("Patch applied.")
PY

# Smoke test
python -c "import sapien; print('sapien import OK')"

# Also install the conda activate.d hook (survives pip reinstalls)
ENV_DIR="$(python -c 'import sys; print(sys.prefix)')"
ACTIVATE_D="$ENV_DIR/etc/conda/activate.d"
DEACTIVATE_D="$ENV_DIR/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

cat > "$ACTIVATE_D/vulkan_icd.sh" <<'HOOK'
#!/usr/bin/env bash
export _CONDA_RF_ORIG_VK_ICD="${VK_ICD_FILENAMES:-__UNSET__}"
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
        :
    elif [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
        export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    fi
fi
HOOK

cat > "$DEACTIVATE_D/vulkan_icd.sh" <<'HOOK'
#!/usr/bin/env bash
if [ "${_CONDA_RF_ORIG_VK_ICD:-__UNSET__}" = "__UNSET__" ]; then
    unset VK_ICD_FILENAMES
else
    export VK_ICD_FILENAMES="$_CONDA_RF_ORIG_VK_ICD"
fi
unset _CONDA_RF_ORIG_VK_ICD
HOOK

echo "activate.d hook installed at $ACTIVATE_D/vulkan_icd.sh"
