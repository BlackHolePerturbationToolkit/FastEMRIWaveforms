#!/usr/bin/env bash
#
# Repair a CUDA plugin wheel into a manylinux wheel, keeping the NVIDIA
# libraries external: they are provided at runtime by the user, through CuPy or
# the nvidia-* wheels, and must not be vendored into the plugin.
#
# The excluded sonames are read from the freshly built extension modules rather
# than hard-coded. NVIDIA soversions do not follow the CUDA major version --
# CUDA 13 ships libcudart.so.13 but kept libcusparse.so.12 -- so deriving them
# from the toolkit version is a reliable way to get it silently wrong: an
# --exclude that matches nothing makes auditwheel vendor the library instead.
#
# Usage: repair-cuda-wheel.sh <wheel> <dest_dir>

set -euo pipefail

wheel="$1"
dest_dir="$2"

# NVIDIA libraries that must stay external, matched on the soname stem only.
# The soversion is whatever the build actually linked against.
#
# Keep this in sync with REQUIRED_LIBS in src/few/cutils/__init__.py. It is a
# superset of REQUIRED_LIBS's library file names (libcublasLt is pulled in
# transitively by libcublas and ships in the same pip component/directory, so
# it has no separate REQUIRED_LIBS entry, but still shows up in a compiled
# .so's NEEDED list and must be excluded here).
external_stems=(libcudart libcublas libcublasLt libcusparse libnvJitLink libnvrtc libcufftw)

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

python -m zipfile -e "${wheel}" "${tmpdir}"

needed=()
while IFS= read -r soname; do
  [[ -n "${soname}" ]] && needed+=("${soname}")
done < <(
  while IFS= read -r -d '' sofile; do
    patchelf --print-needed "${sofile}" 2>/dev/null || true
  done < <(find "${tmpdir}" -type f \( -name '*.so' -o -name '*.so.*' \) -print0) |
    sort -u
)

excluded=()
exclude_args=()
for soname in "${needed[@]}"; do
  for stem in "${external_stems[@]}"; do
    if [[ "${soname}" == "${stem}.so."* ]]; then
      excluded+=("${soname}")
      exclude_args+=(--exclude "${soname}")
      break
    fi
  done
done

# A CUDA plugin always links the CUDA runtime. Finding none means the wheel was
# not built the way we think it was, and repairing it would bundle the whole
# CUDA toolkit -- fail instead.
if [[ ${#excluded[@]} -eq 0 ]]; then
  echo "error: no NVIDIA library found in $(basename "${wheel}")." >&2
  echo "Libraries needed by the wheel were:" >&2
  printf '  %s\n' ${needed[@]+"${needed[@]}"} >&2
  exit 1
fi

echo "Keeping the following NVIDIA libraries external:"
printf '  %s\n' "${excluded[@]}"

auditwheel repair -w "${dest_dir}" "${wheel}" "${exclude_args[@]}"
