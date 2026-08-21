#!/usr/bin/env python3
"""I build a self-contained screening sheet for the three H3 renders."""
import base64, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent

def uri(rel, mime):
    data = (ROOT / rel).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode()

A = {
    "fox_v":   uri("outputs/h3-egpu-test.mp4",  "video/mp4"),
    "bear_v":  uri("outputs/h3-egpu-test2.mp4", "video/mp4"),
    "smoke_v": uri("outputs/smoke-vram.mp4",    "video/mp4"),
    "fox_p":   uri("outputs/web/fox.jpg",   "image/jpeg"),
    "bear_p":  uri("outputs/web/bear.jpg",  "image/jpeg"),
    "smoke_p": uri("outputs/web/smoke.jpg", "image/jpeg"),
}

HTML = """<title>Fox, Bear, Snow</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* A screening sheet is a single committed visual world: I judge footage on a
   dark ground on purpose, so there is no light variant. Every colour below is
   painted explicitly so the page holds on any host background. */
:root{
  --ground:#12171B; --surface:#1A2027; --surface-2:#212A32; --line:#2C3742;
  --ink:#E2E8EB; --muted:#8795A0; --dim:#63707A;
  --amber:#D98A2B;   /* the bear's rim light */
  --ice:#7FA8C4;     /* the snow */
  --ok:#6FA96B;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --disp:"Archivo","Helvetica Neue",Arial,sans-serif;
  --body:"Newsreader",Georgia,"Times New Roman",serif;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--body); font-size:17px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(18px,4vw,40px) 96px}
a{color:var(--ice)}
h1,h2,h3{font-family:var(--disp);text-wrap:balance;margin:0}
h1{font-weight:800;letter-spacing:-.03em;font-size:clamp(2.4rem,6vw,4.1rem);line-height:.98}
h2{font-weight:800;letter-spacing:-.02em;font-size:clamp(1.3rem,2.6vw,1.75rem)}
h3{font-weight:700;letter-spacing:-.01em;font-size:1.02rem}
p{margin:0}
.lede{max-width:64ch;color:var(--muted);font-size:1.14rem;font-weight:300;margin-top:18px}
.lede strong{color:var(--ink);font-weight:400}
.eyebrow{
  font-family:var(--mono);font-size:.7rem;font-weight:600;letter-spacing:.16em;
  text-transform:uppercase;color:var(--dim)
}

/* ---- signal path: host RAM -> Thunderbolt -> VRAM. A real sequence, which is
        the whole story of this build, so it earns the arrow treatment. ---- */
.path{display:flex;flex-wrap:wrap;align-items:stretch;gap:8px;margin:40px 0 0}
.hop{
  flex:1 1 150px;background:var(--surface);border:1px solid var(--line);
  border-radius:3px;padding:12px 14px;display:flex;flex-direction:column;gap:3px
}
.hop .k{font-family:var(--mono);font-size:.66rem;letter-spacing:.14em;text-transform:uppercase;color:var(--dim)}
.hop .v{font-family:var(--disp);font-weight:700;font-size:.96rem}
.hop .n{font-family:var(--mono);font-size:.76rem;color:var(--muted)}
.hop.hot{border-color:#5A4526;background:linear-gradient(180deg,#221B10,var(--surface))}
.hop.hot .v{color:var(--amber)}
.arrow{display:flex;align-items:center;color:var(--dim);font-family:var(--mono);font-size:1.1rem}
@media(max-width:720px){.arrow{display:none}}

section{margin-top:clamp(56px,8vw,92px)}
.shead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
  border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:26px}
.shead p{color:var(--muted);font-size:.95rem;font-weight:300}

/* ---- clips as specimens ---- */
.clips{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:26px}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:4px;overflow:hidden}
figure.lowfi{opacity:.94}
video{display:block;width:100%;height:auto;background:#000}
figcaption{padding:16px 18px 18px}
.ctitle{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:10px}
.badge{
  font-family:var(--mono);font-size:.63rem;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 7px;border-radius:2px;white-space:nowrap;
  border:1px solid currentColor
}
.badge.t{color:var(--ice)}
.badge.d{color:var(--dim)}
.prompt{font-size:.95rem;font-weight:300;color:var(--muted);font-style:italic;margin-bottom:14px}
dl.plate{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;margin:0;
  font-family:var(--mono);font-size:.76rem;border-top:1px solid var(--line);padding-top:12px}
dl.plate dt{color:var(--dim)}
dl.plate dd{margin:0;color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.hint{font-family:var(--mono);font-size:.72rem;color:var(--dim);margin-top:26px;text-align:center}

/* ---- the crux: one bold moment, spent here because it is the finding ---- */
.bw{background:var(--surface);border:1px solid var(--line);border-radius:4px;padding:clamp(20px,3vw,34px)}
.bar{margin-top:22px}
.bar+.bar{margin-top:18px}
.blab{display:flex;justify-content:space-between;align-items:baseline;gap:12px;
  font-family:var(--mono);font-size:.8rem;margin-bottom:7px}
.blab b{font-family:var(--disp);font-size:1.02rem;font-variant-numeric:tabular-nums}
.btrack{height:15px;background:var(--surface-2);border-radius:2px;overflow:hidden}
.bfill{height:100%;border-radius:2px}
.bfill.slow{width:.368%;background:var(--amber);min-width:3px}
.bfill.fast{width:100%;background:linear-gradient(90deg,#2E4A5C,var(--ice))}
.bnote{margin-top:20px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.97rem;font-weight:300;max-width:70ch}
.bnote strong{color:var(--amber);font-weight:400}
.ratio{font-family:var(--disp);font-weight:800;font-size:clamp(2rem,5vw,3rem);
  color:var(--amber);letter-spacing:-.03em;line-height:1;display:block;margin-bottom:4px}

/* ---- data ---- */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:4px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.8rem;min-width:560px}
caption{caption-side:top;text-align:left;padding:14px 16px;color:var(--dim);
  font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  background:var(--surface);border-bottom:1px solid var(--line)}
th,td{padding:9px 16px;text-align:right;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{color:var(--dim);font-weight:600;font-size:.7rem;letter-spacing:.1em;text-transform:uppercase}
tbody tr:last-child td{border-bottom:none}
tr.tot td{color:var(--ink);font-weight:600;background:var(--surface-2)}
td.hi{color:var(--amber)}
.ok{color:var(--ok)}
dl.plate dd.fast{color:var(--amber);font-weight:600}

.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}
.note{background:var(--surface);border:1px solid var(--line);border-left:2px solid var(--ice);
  border-radius:0 4px 4px 0;padding:18px 20px}
.note h3{margin-bottom:8px}
.note p{color:var(--muted);font-size:.95rem;font-weight:300}
.note code,code{font-family:var(--mono);font-size:.82em;color:var(--ice);
  background:var(--surface-2);padding:1px 5px;border-radius:2px}
pre{font-family:var(--mono);font-size:.78rem;background:var(--surface);
  border:1px solid var(--line);border-radius:4px;padding:16px 18px;overflow-x:auto;
  color:var(--muted);line-height:1.65;margin:0}
pre b{color:var(--ink);font-weight:600}
footer{margin-top:88px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--dim);font-family:var(--mono);font-size:.72rem;
  display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
:focus-visible{outline:2px solid var(--ice);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Macmini8,1 &middot; Radeon RX 6900 XT eGPU &middot; MiniMax H3</p>
  <h1>Fox, Bear, Snow</h1>
  <p class="lede">Three clips generated locally by <strong>MiniMax H3</strong> on a
  2018 Intel Mac mini, computed on an external AMD Radeon RX 6900 XT through
  Apple&rsquo;s own Metal driver. No CUDA, no Apple Silicon, no cloud. The model
  is a 31B-parameter DiT whose weights are streamed from SSD two blocks at a
  time, because only <strong>1.678 GiB</strong> of it is ever resident on the GPU.</p>

  <div class="path">
    <div class="hop"><span class="k">Host</span><span class="v">64 GiB DDR4</span><span class="n">i7-8700B &middot; 6C/12T</span></div>
    <div class="arrow">&rarr;</div>
    <div class="hop hot"><span class="k">Thunderbolt 3</span><span class="v">20 Gb/s</span><span class="n">1.76 GB/s measured</span></div>
    <div class="arrow">&rarr;</div>
    <div class="hop"><span class="k">Device</span><span class="v">16 GB VRAM</span><span class="n">gfx1030 &middot; 478 GB/s</span></div>
  </div>
</header>

<section>
  <div class="shead">
    <h2>The renders</h2>
    <p>Unmute any clip &mdash; H3 generates the audio track too.</p>
  </div>
  <div class="clips">

    <figure>
      <video src="__FOX_V__" poster="__FOX_P__" controls loop muted playsinline
             preload="metadata" aria-label="Red fox walking through snow"></video>
      <figcaption>
        <div class="ctitle"><h3>Red fox</h3><span class="badge t">Target &middot; seed 42</span></div>
        <p class="prompt">&ldquo;A red fox walks through fresh snow in a pine forest.
        Medium tracking shot, natural winter light, realistic fur, soft footsteps
        in snow and gentle wind in the trees.&rdquo;</p>
        <dl class="plate">
          <dt>Resolution</dt><dd>608 &times; 352</dd>
          <dt>Frames / steps</dt><dd>22 / 4</dd>
          <dt>DiT layers</dt><dd>50 of 50</dd>
          <dt>Generation time</dt><dd class="fast">6 min 4 s</dd>
          <dt>Peak host RAM</dt><dd>15.46 GB</dd>
          <dt>Swap</dt><dd class="ok">0</dd>
          <dt>Audio</dt><dd class="ok">AAC, 30 frames</dd>
        </dl>
      </figcaption>
    </figure>

    <figure>
      <video src="__BEAR_V__" poster="__BEAR_P__" controls loop muted playsinline
             preload="metadata" aria-label="Brown bear in a rushing river"></video>
      <figcaption>
        <div class="ctitle"><h3>Brown bear</h3><span class="badge t">Target &middot; seed 1234</span></div>
        <p class="prompt">&ldquo;A brown bear catches a salmon in a rushing mountain
        river. Close tracking shot, golden late-afternoon light, water spray in
        the air, sound of rushing water and splashing.&rdquo;</p>
        <dl class="plate">
          <dt>Resolution</dt><dd>608 &times; 352</dd>
          <dt>Frames / steps</dt><dd>22 / 4</dd>
          <dt>DiT layers</dt><dd>50 of 50</dd>
          <dt>Generation time</dt><dd>64 min 53 s</dd>
          <dt>Peak host RAM</dt><dd>35.34 GB</dd>
          <dt>Swap</dt><dd class="ok">0</dd>
          <dt>Audio</dt><dd class="ok">AAC, 30 frames</dd>
        </dl>
      </figcaption>
    </figure>

    <figure class="lowfi">
      <video src="__SMOKE_V__" poster="__SMOKE_P__" controls loop muted playsinline
             preload="metadata" aria-label="Low-step diagnostic render of a fox in snow"></video>
      <figcaption>
        <div class="ctitle"><h3>First light</h3><span class="badge d">Diagnostic</span></div>
        <p class="prompt">The run that first proved the eGPU could finish a
        generation at all. Deliberately starved: a third of the resolution, 2
        denoising passes instead of 4, 35 DiT blocks instead of 50. Structure is
        right, detail is not &mdash; which is exactly what those settings buy.</p>
        <dl class="plate">
          <dt>Resolution</dt><dd>256 &times; 160</dd>
          <dt>Frames / steps</dt><dd>22 / 2</dd>
          <dt>DiT layers</dt><dd>35 of 50</dd>
          <dt>Generation time</dt><dd>5 min 43 s</dd>
          <dt>Peak host RAM</dt><dd>33.81 GB</dd>
          <dt>Swap</dt><dd class="ok">0</dd>
          <dt>Audio</dt><dd class="ok">AAC, 30 frames</dd>
        </dl>
      </figcaption>
    </figure>

  </div>
  <p class="hint">All three clips are 0.92 s at 24 fps &mdash; they loop.</p>
</section>

<section>
  <div class="shead">
    <h2>Why it nearly didn&rsquo;t work</h2>
    <p>One number explains the entire port.</p>
  </div>
  <div class="bw">
    <p class="eyebrow">GPU read bandwidth by Metal storage mode</p>
    <div class="bar">
      <div class="blab"><span>Shared &mdash; host memory across Thunderbolt</span><b>1.76 GB/s</b></div>
      <div class="btrack"><div class="bfill slow"></div></div>
    </div>
    <div class="bar">
      <div class="blab"><span>Private &mdash; on-card VRAM</span><b>478 GB/s</b></div>
      <div class="btrack"><div class="bfill fast"></div></div>
    </div>
    <p class="bnote"><span class="ratio">270&times;</span>
    That sliver is drawn to scale. h3.c allocates every buffer
    <code>MTLResourceStorageModeShared</code>, which is free on Apple Silicon
    where memory is unified &mdash; and catastrophic on an external GPU, where it
    means host memory reached over a cable. Its matmul reloads the weight tile
    for every 16 rows of output, so a <strong>294 MiB</strong> FC1 weight became
    tens of GiB of cable traffic and a single dispatch ran <strong>23 s</strong>.
    macOS kills anything that stalls the GPU: <code>checkGPUProgress</code> fired,
    the driver restarted the channel twice, then refused every further submission
    from the process. Staging those weights into VRAM took the same dispatch to
    <strong>0.25 s</strong>.</p>
  </div>
</section>

<section>
  <div class="shead"><h2>Where the 51 minutes go</h2><p>Red fox run, per phase.</p></div>
  <div class="scroll">
  <table>
    <caption>Phase profile &mdash; 608&times;352, 22 frames, 4 steps, 50 layers &mdash; before and after</caption>
    <thead><tr><th>Phase</th><th>Wall before</th><th>Wall after</th><th>Faster</th><th>GPU after</th></tr></thead>
    <tbody>
      <tr><td>Qwen text encoder</td><td>147.5 s</td><td>38.9 s</td><td class="hi">3.8&times;</td><td>3.55 s</td></tr>
      <tr><td>DiT load</td><td>38.6 s</td><td>41.8 s</td><td>0.9&times;</td><td>0.40 s</td></tr>
      <tr><td>DiT Euler denoise</td><td>1934.1 s</td><td>262.3 s</td><td class="hi">7.4&times;</td><td>9.26 s</td></tr>
      <tr><td>Audio VAE decode</td><td>2.6 s</td><td>1.2 s</td><td>2.2&times;</td><td>0.44 s</td></tr>
      <tr><td>Video VAE decode</td><td>956.3 s</td><td>18.3 s</td><td class="hi">52.4&times;</td><td>3.18 s</td></tr>
      <tr class="tot"><td>Total</td><td>3081.6 s</td><td>364.2 s</td><td class="hi">8.46&times;</td><td>&mdash;</td></tr>
    </tbody>
  </table>
  </div>
  <p class="hint">Output after optimization is <strong>bit-identical</strong> to
  the run before it &mdash; md5 886b40d1&hellip; both times. Pure speed, no quality
  trade. The GPU column counts root command buffers only, so it is a floor.</p>
</section>

<section>
  <div class="shead"><h2>What changed in the runtime</h2>
    <p>Five commits, 520 insertions, all discrete-GPU paths gated so Apple Silicon is untouched.</p></div>
  <div class="notes">
    <div class="note">
      <h3>Explicit GPU selection</h3>
      <p>Upstream calls <code>MTLCreateSystemDefaultDevice()</code>, correct where
      there is exactly one GPU. This Mac has two. The selector honours
      <code>H3_METAL_DEVICE_NAME</code>, <code>_INDEX</code> and
      <code>_REGISTRY_ID</code>, and returns nothing rather than quietly running a
      31B video model on an Intel UHD 630.</p>
    </div>
    <div class="note">
      <h3>Weights in VRAM</h3>
      <p>Streamed DiT matrices are allocated private with a host-visible staging
      buffer; the SSD read lands in staging and blits across. Two slots, about
      770 MiB each &mdash; comfortable inside 16 GiB.</p>
    </div>
    <div class="note">
      <h3>Bounded command buffers</h3>
      <p>Work is flushed every few operations with a cap on buffers in flight, so
      no single submission outlives the watchdog. Metal runs same-queue buffers in
      commit order, so the arithmetic is unchanged.</p>
    </div>
    <div class="note">
      <h3>Activations in VRAM</h3>
      <p>Every intermediate tensor was in host memory too, so it crossed the
      cable twice &mdash; written by one kernel, read by the next. Moving them to
      private VRAM with lazy staging for CPU access took the video VAE decode
      from 956 s to 18 s.</p>
    </div>
    <div class="note">
      <h3>A 24.2 GiB pool leak</h3>
      <p>Metal returns buffers through the autorelease pool. The AdaLN precompute
      is a plain-C loop with no pool, so all 50 of its 496 MiB projection weights
      stayed resident for the whole run. One pool per iteration gave back
      24.2 GiB.</p>
    </div>
    <div class="note">
      <h3>No dtype port needed</h3>
      <p>All 145 native <code>bfloat</code> uses sit behind an M5-only guard. The
      portable kernels keep BF16 as <code>ushort</code> and convert by bit
      manipulation, accumulating in float &mdash; so no BF16 hardware is required.
      MPSGraph BF16 matmul came back bit-exact on gfx1030.</p>
    </div>
  </div>
</section>

<section>
  <div class="shead"><h2>Reproduce</h2><p>From the patched tree.</p></div>
<pre><b>./generate.sh</b> "A red fox walks through fresh snow in a pine forest. \\
Medium tracking shot, natural winter light, realistic fur, \\
soft footsteps in snow and gentle wind in the trees." \\
  <b>608 352 22 4</b> h3-egpu-test.mp4 <b>42</b>

I found the RX 6900 XT: AMD Radeon RX 6900 XT
I selected Metal GPU:
  name                  = AMD Radeon RX 6900 XT
  architecture          = amdgpu_gfx1030
  recommendedWorkingSet = 15.98 GiB
I am streaming the BF16 DiT layers from SSD.
<b>I saved outputs/h3-egpu-opt.mp4 (375711 bytes) in 364s.</b></pre>
</section>

<footer>
  <span>MiniMax H3 &middot; antirez/h3.c @ 8974cc0 + branch intel-amd-egpu</span>
  <span>macOS 15.7.9 &middot; Metal 3 &middot; 2026-08-21</span>
</footer>
</div>

<script>
/* Autoplay the loops, but never against a stated preference for less motion. */
(function () {
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  document.querySelectorAll("video").forEach(function (v) {
    if (calm) return;
    v.autoplay = true;
    var go = v.play();
    if (go && go.catch) go.catch(function () { /* fine: the poster stands in */ });
  });
})();
</script>
"""
for k, v in A.items():
    HTML = HTML.replace("__" + k.upper() + "__", v)
out = ROOT / "outputs/web/renders.html"
out.write_text(HTML)
print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
