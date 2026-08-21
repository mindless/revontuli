# Linux / ROCm fallback for MiniMax H3 on the RX 6900 XT

**Status: PREPARED, NOT EXECUTED. This path was not needed.**

The macOS Metal path (Path A) cleared every capability gate on this machine —
see `../report.md`. This document exists so the Linux route is ready if the
macOS path later regresses, and so the risks are written down honestly rather
than discovered at 2 a.m. with a wiped disk.

Nothing in this document has been run. No disk was erased, no ISO was written,
no NVRAM boot argument was changed, no security policy was altered. SIP is still
`enabled` and `systemextensionsctl list` still reports 0 extensions.

---

## Why this is a fallback and not the primary path

| | macOS + h3.c | Linux + ROCm |
|---|---|---|
| Driver | Apple's own signed `AMDRadeonX6000` (already working) | `amdgpu`, but only after a **custom kernel module patches the PCI resource tree** |
| Risk to the machine | none taken | requires erasing an external disk and relaxing Startup Security |
| H3 implementation | native, purpose-built, SSD-streaming, already builds and passes 1769 tests here | ComfyUI or SGLang, neither validated on 16 GiB RDNA2 |
| 33B model residency | solved by `--ssd-streaming` (~2.0 GiB tracked DiT storage) | depends on layerwise CPU offload that must be re-verified |
| VRAM headroom | 15.98 GiB working set, and Metal will over-commit and page | 16 GiB hard, and upstream reports OOM on 16 GiB Radeon |

---

## Verified facts (checked 2026-08-21, not recalled)

**ROCm 7.2.4 / 7.2.3 compatibility matrix**
(`https://rocm.docs.amd.com/en/docs-7.2.4/compatibility/compatibility-matrix.html`)

- `gfx1030` (RDNA2, Navi 21 — this card) **is** an officially listed LLVM target.
- Supported operating systems: **Ubuntu 24.04.4** and **Ubuntu 22.04.5**.

So the userspace stack is officially available for this GPU. The hard part is
not ROCm; it is getting `amdgpu` to own the card at all behind a T2 Mac's
Thunderbolt bridge.

**`philmcneely/t2-egpu-linux`** @ `68bdab3` (2026-07-26)

This project is unusually relevant: it targets **the same Mac model**.

- Target host: **Mac Mini 2018 (Macmini8,1) with T2** — identical to this machine.
- Target GPU: **AMD Radeon RX 6800** (Navi 21, gfx1030). The README says
  explicitly: *"AMD Radeon RX 6800 (Navi 21, gfx1030) — **NOT the XT**"*.
- Target kernel: **7.0.9-1-t2-noble** (Ubuntu 24.04).
- Root cause it solves: *"The T2 Mac Mini's firmware allocates only 224MB of
  prefetchable memory for the entire Thunderbolt domain. The RX 6800's BAR 0
  needs at minimum 256MB."*
- What `egpu_bar.c` does: it does **not** resize BARs. It patches the kernel's
  internal `struct resource` for BAR 0 and reparents it to the bridge's
  prefetchable window so the kernel's view matches the registers that `setpci`
  already programmed.

Its hardcoded assumptions, verbatim from `egpu_bar.c`:

```c
#define GPU_VENDOR  0x1002
#define GPU_DEVICE  0x73bf
#define BAR0_ADDR   0x4010000000ULL
#define BAR0_SIZE   0x10000000ULL       /* 256MB */
#define PREF_START  0x4010000000ULL
#define PREF_END    (0x4010000000ULL + 0x10000000ULL - 1)
```

**Its explicit warnings, which I am carrying forward rather than paraphrasing:**

- *"NEVER do a full PCI rescan."* `echo 1 > /sys/bus/pci/rescan` — *"it crashes
  the machine every time. EVERY time."*
- `amdgpu.rebar=0` is **critical**. Without it, amdgpu auto-resizes BAR 0 from
  256 MB to 16 GB, overflows the bridge windows, and causes a kernel BUG with
  GART page-table corruption.
- `pcie_aspm=off` is needed to keep the Thunderbolt link stable.

---

## Assumption comparison: the repo vs THIS machine

Everything in the "my machine" column was measured on macOS and is recorded in
`../logs/`. The Linux-side rows genuinely cannot be filled in until Linux boots,
and I have marked them `UNKNOWN` rather than guessing.

| Assumption | t2-egpu-linux | My machine | Match? |
|---|---|---|---|
| Mac model | Mac Mini 2018, Macmini8,1, T2 | **Macmini8,1**, T2 | **YES** |
| CPU | Intel (Coffee Lake) | Intel i7-8700B | **YES** |
| GPU family | Navi 21 / RDNA2 / gfx1030 | Navi 21 / RDNA2 / **`amdgpu_gfx1030`** (from Metal) | **YES** |
| GPU SKU | RX 6800, "**NOT the XT**" | RX **6900 XT** | **DIFFERENT SKU** — same silicon, higher bin |
| PCI vendor:device | `1002:73bf` | **`0x1002:0x73bf`** (rev `0xc0`) | **YES** — Navi 21 shares 0x73BF across 6800/6800 XT/6900 XT |
| GPU VRAM | 16 GB | 16 GB | **YES** |
| Enclosure | AKiTiO Node Titan | **Razer Core X Chroma** (Vendor ID 0x127, FW 40.1) | **DIFFERENT** — bridge topology may differ |
| TB link speed | not stated | **20 Gb/s** negotiated upstream (not 40) | **UNKNOWN** |
| TB bus / port | not stated | Bus 0, Receptacle 1; enclosure presents 2 chained TB devices (Route String 3 and 103) | **UNKNOWN** |
| Bridge prefetch window | 224 MB from T2 firmware | **UNKNOWN** until `lspci -vv` under Linux | **MUST VERIFY** |
| BAR0 requirement | 256 MB | expected 256 MB (same silicon) | likely, **MUST VERIFY** |
| BAR0 target address | hardcoded `0x4010000000` | **UNKNOWN** — depends on this enclosure's bridge chain | **MUST VERIFY — this is the blocking unknown** |
| Kernel | 7.0.9-1-t2-noble (Ubuntu 24.04) | n/a (macOS 15.7.9) | n/a |
| ROCm gfx1030 support | assumed | **confirmed** in ROCm 7.2.4 matrix | **YES** |

### Verdict on loading the module

**DO NOT load `egpu_bar.ko` yet.** Two rows above are hard blockers:

1. **`BAR0_ADDR` is a hardcoded literal**, not discovered at runtime. It was
   derived from the author's AKiTiO Node Titan on his port. This machine uses a
   Razer Core X Chroma that presents a *chained* pair of Thunderbolt devices
   (Route String 3 and 103), so the bridge chain — and therefore the address the
   firmware programs — may well differ. Patching the kernel's resource tree to
   point at the wrong physical address is exactly how you get the GART
   corruption the README warns about.
2. **The enclosure differs**, which is the component that determines the bridge
   topology the module assumes.

The correct sequence is: boot Linux, run `linux_postinstall.sh`, read the real
`lspci -vv` BAR assignments and bridge windows, and only then decide whether
`BAR0_ADDR` must be changed to a machine-specific value. That is a
**HIGH-RISK MANUAL CHECKPOINT** requiring explicit human approval.

---

## Manual checkpoint 1 — writing the installer USB (NOT EXECUTED)

`dd` against a physical device is destructive and irreversible, so I stop here.

Current external physical disks on this machine, for identification only:

```
$ diskutil list external physical
```

I did run the read-only listing (saved as `external-disks-snapshot.txt`). Every
external physical disk currently attached **holds data**:

| Disk | Size | Contents |
|---|---|---|
| `/dev/disk2` | 500.1 GB | EFI + Apple_APFS container disk3 |
| `/dev/disk4` | 3.0 TB | EFI + Apple_APFS container disk8 |
| `/dev/disk5` | 3.0 TB | EFI + Apple_APFS containers disk10, disk9 |
| `/dev/disk6` | 1000.0 GB | EFI + Apple_APFS container disk12 |
| `/dev/disk7` | 119.8 GB | EFI + Apple_APFS container disk11 |
| `/dev/disk13` | 2.0 TB | EFI + Microsoft Basic Data "RIMFIRE" |

**There is no obviously disposable drive here.** Several of these are Time
Machine targets (`/Volumes/Tardis` holds dated backups). I am therefore not
going to nominate a device, even as an example with a real identifier.

Run that yourself and identify a **disposable** installer drive. Then, and only
after you are certain of the identifier:

```bash
# EXAMPLE ONLY — I have NOT run this. Replace diskN with YOUR disposable drive.
# Getting this wrong destroys the wrong disk. There is no undo.
diskutil unmountDisk /dev/diskN
sudo dd if=<T2-UBUNTU.iso> of=/dev/rdiskN bs=4m status=progress
```

Get the ISO from the T2 Linux project and verify whatever checksum/signature the
project publishes for that exact release:

- `https://github.com/t2linux/T2-Ubuntu` (release ISOs)
- `https://github.com/t2linux/T2-Debian-and-Ubuntu-Kernel` (current kernel)

Pick an Ubuntu release that is in **both** the T2 Linux supported list **and**
the ROCm matrix above. As of 2026-08-21 that intersection points at
**Ubuntu 24.04**, but re-check both projects at install time rather than trusting
this line.

## Manual checkpoint 2 — Startup Security Utility (NOT AUTOMATED)

T2 Macs will not boot an external or non-Apple OS until you allow it by hand:

1. Shut down. Power on holding **Cmd-R** to enter Recovery.
2. **Utilities → Startup Security Utility.**
3. Authenticate as an administrator.
4. Set **Secure Boot** to *Medium Security* or *No Security* as the T2 Linux
   documentation for your chosen release requires.
5. Set **External Boot** to *Allow booting from external media*.

I have deliberately not automated any part of this. It is a security posture
change and it should be a conscious, physical act.

## Manual checkpoint 3 — the BAR module (NOT EXECUTED, HIGH RISK)

Only after `linux_postinstall.sh` has shown you the real topology. Back up every
file before editing it:

```bash
sudo cp /etc/default/grub /etc/default/grub.bak.$(date +%s)
# and any of these that already exist:
sudo cp /etc/modprobe.d/amdgpu-egpu.conf   /etc/modprobe.d/amdgpu-egpu.conf.bak   2>/dev/null
sudo cp /usr/local/bin/egpu-init.sh        /usr/local/bin/egpu-init.sh.bak        2>/dev/null
sudo cp /etc/systemd/system/egpu-init.service \
        /etc/systemd/system/egpu-init.service.bak 2>/dev/null
```

Kernel parameters the upstream project requires (do not cargo-cult these
without reading its README for your kernel version):

```
amdgpu.rebar=0 pcie_aspm=off
```

And the standing prohibition, worth repeating because it is the one mistake that
takes the whole machine down:

```bash
# NEVER RUN THIS on this hardware.
# echo 1 > /sys/bus/pci/rescan
```

---

## If Linux does come up: the H3 runtime question

Be honest about 16 GiB. Detected system RAM here is **64.00 GiB**, which is the
saving grace — but VRAM is 16 GiB and that is the binding constraint.

Investigate in this order, re-verifying each claim at execution time:

1. **Native ComfyUI MiniMax H3 under ROCm.** Confirm the checked-out commit
   really contains `comfy.ldm.minimax.model.MiniMaxH3Model` (or its current
   equivalent) before installing anything. Use `python main.py --help` to find
   the *current* low-VRAM/offload flags; do not invent stale ones.
2. **SGLang Diffusion MiniMax H3 under ROCm**, following the current cookbook at
   `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`.
3. A current H3-specific ROCm implementation with primary-source evidence.

Do not substitute another video model. Do not port NVIDIA-only FlashAttention,
CUDA graphs, or FP8 paths onto ROCm.

**Known 16 GiB risk signals to re-check, not assume:**
- the verified SGLang single-consumer-GPU recipe is described as 24 GB-class,
  peaking near 18 GB with offload
- AMD's own SGLang validation is on MI300X/MI355X, not consumer RDNA2
- ROCm issue #6567 reports OOM on a 16 GB Radeon
- a 32 GB Radeon success report exists (Spectrum-H3 issue #6), which is a
  different memory class
- ROCm issue #6123 reports **silent CPU→GPU NaN corruption on gfx1030**

That last one matters most. `linux_postinstall.sh` therefore includes an
aggressive ≥100-iteration transfer-correctness test. A 33B video model requires
trustworthy transfers, and corruption must never be papered over with retries.

Do not declare 16 GiB impossible merely from a static requirement if the runtime
genuinely supports layerwise CPU offload. Do not declare it viable merely
because the model loads. The bar is a coherent MP4.
