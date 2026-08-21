#!/usr/bin/env bash
# linux_postinstall.sh
#
# I run this as the FIRST thing after manually booting the external Linux
# installation on the Mac mini 2018. I am read-only and non-destructive: I
# collect facts, compare them against what t2-egpu-linux assumes, and tell you
# whether it is safe to go further. I do not load kernel modules, I do not edit
# system files, and I never touch /sys/bus/pci/rescan.
#
# Usage:
#   ./linux_postinstall.sh            # diagnostics only
#   ./linux_postinstall.sh --rocm     # also probe an installed ROCm stack
#   ./linux_postinstall.sh --torch    # also run the transfer-corruption test
#
set -uo pipefail

# ---------------------------------------------------------------------------
# I refuse to run anywhere except Linux, so that accidentally invoking this on
# macOS does nothing at all.
# ---------------------------------------------------------------------------
if [ "$(uname -s)" != "Linux" ]; then
    echo "I only run on Linux. This is $(uname -s), so I am exiting safely."
    echo "I am meant to be run after you have manually booted the external"
    echo "Linux installation. Nothing has been changed."
    exit 0
fi

OUT="${1:-$HOME/minimax-h3-egpu-linux-logs}"
case "$OUT" in --*) OUT="$HOME/minimax-h3-egpu-linux-logs" ;; esac
mkdir -p "$OUT"
echo "I am writing diagnostics to $OUT"

DO_ROCM=0; DO_TORCH=0
for arg in "$@"; do
    case "$arg" in
        --rocm)  DO_ROCM=1 ;;
        --torch) DO_TORCH=1 ;;
    esac
done

run() {  # run <logfile> <description> <command...>
    local log="$OUT/$1"; shift
    local desc="$1"; shift
    printf '  %-46s' "$desc"
    if "$@" > "$log" 2>&1; then echo "ok -> $(basename "$log")"
    else echo "(non-zero exit; see $(basename "$log"))"; fi
}

echo
echo "==================== 1. host identity ===================="
run uname.txt        "uname -a"                 uname -a
run os-release.txt   "/etc/os-release"          cat /etc/os-release
run lscpu.txt        "lscpu"                    lscpu
run free.txt         "free -h"                  free -h
run lsblk.txt        "lsblk"                    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL
run dmidecode.txt    "board/product name"       sh -c 'cat /sys/class/dmi/id/product_name /sys/class/dmi/id/board_name 2>/dev/null'

echo
echo "  kernel : $(uname -r)"
echo "  distro : $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}")"
echo "  RAM    : $(awk '/MemTotal/{printf "%.2f GiB", $2/1048576}' /proc/meminfo)"
if uname -r | grep -q t2; then
    echo "  I am running a T2-patched kernel ($(uname -r)). Good."
else
    echo "  WARNING: this kernel name does not look T2-patched. Internal"
    echo "           keyboard/audio/Thunderbolt behaviour may be wrong."
fi

echo
echo "==================== 2. PCI topology ===================="
run lspci-nn.txt     "lspci -nn"                lspci -nn
run lspci-tv.txt     "lspci -tv"                lspci -tv
run lsmod.txt        "lsmod"                    lsmod
run thunderbolt.txt  "thunderbolt devices"      sh -c 'ls -la /sys/bus/thunderbolt/devices 2>/dev/null || echo "no thunderbolt bus"'
run dmesg-gpu.txt    "dmesg amdgpu/tb/pcie/bar" sh -c "dmesg -T 2>/dev/null | grep -Ei 'amdgpu|thunderbolt|pcie|bar|iommu|drm' || true"

# I locate the GPU by vendor:device rather than by name, because the Navi 21
# device ID 0x73bf is shared by the RX 6800, 6800 XT and 6900 XT.
GPU_BDF="$(lspci -nn 2>/dev/null | grep -i '\[1002:73bf\]' | awk '{print $1}' | head -1)"
if [ -z "$GPU_BDF" ]; then
    # Fall back to any AMD VGA/Display controller, in case the ID differs.
    GPU_BDF="$(lspci -nn 2>/dev/null \
               | grep -Ei 'VGA|Display|3D controller' \
               | grep -i '1002:' | awk '{print $1}' | head -1)"
fi

if [ -z "$GPU_BDF" ]; then
    echo
    echo "  I did NOT find an AMD GPU on the PCI bus."
    echo "  I am NOT going to run 'echo 1 > /sys/bus/pci/rescan'. The"
    echo "  t2-egpu-linux project reports that it crashes this hardware every"
    echo "  time. Reboot with the enclosure powered on before boot instead."
    echo
    echo "STOP: no GPU to analyse. Diagnostics are in $OUT"
    exit 3
fi

echo
echo "  I found an AMD GPU at PCI $GPU_BDF:"
lspci -nn -s "$GPU_BDF" | sed 's/^/    /'
run "lspci-vv-$GPU_BDF.txt" "lspci -vv on the GPU" lspci -vv -s "$GPU_BDF"

echo
echo "  --- what actually matters, extracted ---"
{
    echo "### lspci -vv -s $GPU_BDF (filtered)"
    lspci -vv -s "$GPU_BDF" 2>/dev/null | grep -Ei \
        'Region|Kernel driver in use|Kernel modules|LnkCap|LnkSta|Memory at|prefetchable|Capabilities: .*Atomic'
} | tee "$OUT/gpu-key-facts.txt" | sed 's/^/    /'

DRIVER="$(lspci -k -s "$GPU_BDF" 2>/dev/null | awk -F': ' '/Kernel driver in use/{print $2}')"
echo
echo "  kernel driver in use : ${DRIVER:-<none>}"

echo
echo "  --- bridge chain above the GPU (BAR windows live here) ---"
BRIDGE_PATH="$(readlink -f "/sys/bus/pci/devices/0000:$GPU_BDF" 2>/dev/null || true)"
if [ -n "$BRIDGE_PATH" ]; then
    echo "$BRIDGE_PATH" | tr '/' '\n' | grep -E '^[0-9a-f]{4}:' \
        | while read -r dev; do
            short="${dev#0000:}"
            desc="$(lspci -s "$short" 2>/dev/null | cut -d' ' -f2-)"
            win="$(lspci -vv -s "$short" 2>/dev/null \
                   | grep -Ei 'Memory behind bridge|Prefetchable memory behind bridge' \
                   | tr -s ' ' | paste -sd' ' -)"
            printf '    %-12s %s\n' "$short" "${desc:0:60}"
            [ -n "$win" ] && printf '        %s\n' "$win"
          done | tee "$OUT/bridge-chain.txt"
else
    echo "    (could not resolve the sysfs path)"
fi

echo
echo "  --- IOMMU state ---"
if [ -d /sys/kernel/iommu_groups ] && [ -n "$(ls -A /sys/kernel/iommu_groups 2>/dev/null)" ]; then
    echo "    IOMMU is ACTIVE ($(ls /sys/kernel/iommu_groups | wc -l) groups)"
else
    echo "    IOMMU appears inactive or not exposed"
fi
grep -o 'iommu=[^ ]*\|amd_iommu=[^ ]*\|intel_iommu=[^ ]*\|amdgpu.rebar=[^ ]*\|pcie_aspm=[^ ]*' \
    /proc/cmdline 2>/dev/null | sed 's/^/    cmdline: /' || true

echo
echo "  --- render nodes ---"
if ls /dev/dri/renderD* >/dev/null 2>&1; then
    ls -la /dev/dri/ | sed 's/^/    /'
    echo "    amdgpu appears to own a render node. This is the gate for ROCm."
else
    echo "    NO /dev/dri/renderD* present."
    echo "    ROCm cannot work until amdgpu owns the GPU and exposes a render node."
fi

echo
echo "==================== 3. safety comparison ===================="
cat <<'NOTE'
  Before anyone considers building or loading t2-egpu-linux's egpu_bar.ko,
  compare the numbers above against what that module HARDCODES:

      GPU_VENDOR  0x1002
      GPU_DEVICE  0x73bf
      BAR0_ADDR   0x4010000000      <-- a fixed literal, not discovered
      BAR0_SIZE   0x10000000        (256 MB)

  That address came from the author's AKiTiO Node Titan enclosure. This machine
  uses a Razer Core X Chroma with a CHAINED pair of Thunderbolt devices, so the
  bridge chain and the firmware-programmed address may differ.

  Check, in gpu-key-facts.txt and bridge-chain.txt above:
    * Does the GPU's Region 0 actually sit at 0x4010000000?
    * Is the prefetchable window behind the bridge large enough (>= 256 MB)?
    * Does the bridge chain have the same depth the module walks?

  If ANY of those differ, STOP. Patching the kernel resource tree to the wrong
  physical address is precisely how the GART page-table corruption in that
  project's README happens.

  And never, on this hardware:
      echo 1 > /sys/bus/pci/rescan
NOTE

if [ "$DO_ROCM" = "1" ]; then
    echo
    echo "==================== 4. ROCm stack ===================="
    for tool in /opt/rocm/bin/rocminfo rocminfo rocm-smi amd-smi clinfo; do
        if command -v "$tool" >/dev/null 2>&1 || [ -x "$tool" ]; then
            base="$(basename "$tool")"
            run "rocm-$base.txt" "$base" "$tool"
        fi
    done
    if [ -r /opt/rocm/.info/version ]; then
        echo "  ROCm version: $(cat /opt/rocm/.info/version)"
    fi
    # I report the gfx target, because ROCm silently mis-JITs on a mismatch.
    if command -v rocminfo >/dev/null 2>&1; then
        echo "  gfx targets: $(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | sort -u | paste -sd' ' -)"
    fi
fi

if [ "$DO_TORCH" = "1" ]; then
    echo
    echo "==================== 5. transfer-corruption test ===================="
    echo "  ROCm issue #6123 reports silent CPU->GPU NaN corruption on"
    echo "  gfx1030. A 33B video model cannot tolerate that, so I test it hard."
    PY="$(command -v python3 || true)"
    if [ -z "$PY" ]; then
        echo "  I could not find python3; skipping."
    else
        "$PY" - <<'PYEOF'
import sys
try:
    import torch
except Exception as exc:
    print(f"  torch import failed: {exc}")
    sys.exit(0)

print(f"  torch            = {torch.__version__}")
print(f"  torch.version.hip= {getattr(torch.version, 'hip', None)}")
print(f"  cuda.is_available= {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  No HIP device visible to torch; stopping.")
    sys.exit(0)
print(f"  device name      = {torch.cuda.get_device_name(0)}")
print(f"  device props     = {torch.cuda.get_device_properties(0)}")

dev = torch.device("cuda:0")
ITERS = 100
sizes = [(256, 256), (1024, 1024), (4096, 4096), (8192, 1024)]
dtypes = [torch.float32, torch.float16]
if torch.cuda.is_bf16_supported():
    dtypes.append(torch.bfloat16)
else:
    print("  note: bf16 not reported as supported; skipping bf16")

failures = 0
for dtype in dtypes:
    for shape in sizes:
        # A FIXED reference tensor, transferred repeatedly. Any variation
        # between iterations is corruption, not numerics.
        ref = torch.randn(*shape, dtype=torch.float32)
        ref_t = ref.to(dtype)
        bad_nan = bad_roundtrip = bad_mm = bad_linear = 0
        for _ in range(ITERS):
            g = ref_t.to(dev, non_blocking=False)
            torch.cuda.synchronize()
            if torch.isnan(g.float()).any() or torch.isinf(g.float()).any():
                bad_nan += 1
            back = g.cpu()
            if not torch.equal(back, ref_t):
                bad_roundtrip += 1
            if shape[0] == shape[1]:
                mm = (g @ g).float().cpu()
                if torch.isnan(mm).any():
                    bad_mm += 1
                lin = torch.nn.functional.linear(g, g).float().cpu()
                if torch.isnan(lin).any():
                    bad_linear += 1
            torch.cuda.synchronize()
        tag = str(dtype).replace("torch.", "")
        status = "OK"
        if bad_nan or bad_roundtrip or bad_mm or bad_linear:
            status = "*** CORRUPTION ***"
            failures += 1
        print(f"  {tag:>9} {str(shape):>14}  nan={bad_nan:<4} "
              f"roundtrip_mismatch={bad_roundtrip:<4} mm_nan={bad_mm:<4} "
              f"linear_nan={bad_linear:<4} {status}")

print()
if failures:
    print("  VERDICT: I reproduced nondeterministic corruption. I mark this")
    print("           ROCm/PyTorch combination UNSAFE for MiniMax H3. Do not")
    print("           mask this with retries -- fix or change the stack.")
    sys.exit(1)
print("  VERDICT: no corruption observed across "
      f"{ITERS} iterations per configuration.")
PYEOF
    fi
fi

echo
echo "==================== summary ===================="
echo "  GPU PCI BDF      : $GPU_BDF"
echo "  kernel driver    : ${DRIVER:-<none>}"
echo "  render node      : $(ls /dev/dri/renderD* 2>/dev/null | paste -sd' ' - || echo none)"
echo "  logs             : $OUT"
echo
echo "  I changed nothing on this system. Next step is a HUMAN decision:"
echo "  read $OUT/gpu-key-facts.txt and $OUT/bridge-chain.txt, compare them to"
echo "  the hardcoded values in LINUX_EXTERNAL_SETUP.md, and only then decide"
echo "  whether the BAR module is applicable to THIS enclosure."
