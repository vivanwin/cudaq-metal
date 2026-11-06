# shellcheck shell=bash
# Sourced by the build scripts (not run directly).
#
# cudaq_env PYTHON — locate the cudaq installed for PYTHON without importing it
# and set CUDAQ_VERSION (source tag, e.g. 0.16.0) and CUDAQ_LIB_DIR
# (<site-packages>/lib). Two separate reads, so paths with spaces survive.
# shellcheck disable=SC2034  # both are read by the sourcing script
cudaq_env() {
  { read -r CUDAQ_VERSION; read -r CUDAQ_LIB_DIR; } < <("$1" - <<'EOF'
import importlib.metadata as md, importlib.util, os, re
spec = importlib.util.find_spec("cudaq")
if spec is None or not spec.origin:
    raise SystemExit("cudaq is not installed for this Python")
print(re.match(r"\d+\.\d+\.\d+", md.version("cuda-quantum-cu13")).group(0))
print(os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "lib"))
EOF
  )
}
