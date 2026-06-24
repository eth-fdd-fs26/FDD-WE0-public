"""Presentation & quiz helpers for the Block 3 PyTorch-debugging notebook.

Same idea as `pdm_viz` for Block 2: all the HTML/CSS illustrations and the quiz
*answer keys* live here, out of the notebook, so the teaching cells stay about
PyTorch and the quizzes can't be solved by reading the cell. The notebook does::

    import torch_viz
    torch_viz.water_pipe_analogy()
    torch_viz.mc_quiz("fp16_speedup")

Students are told not to read this file.
"""
import json as _json

from IPython.display import HTML, display


# ===========================================================================
#  Generic quiz renderers  (mirrors pdm_viz: mc / true-false / numeric)
# ===========================================================================
def _mc_render(title, question, options, answer_index, reveal):
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = "mc_" + str(abs(hash((question, tuple(options), answer_index))) % 10**8)
    rows = "".join(
        '<div class="mc-opt" data-i="%d"><span class="mc-dot"></span>%s</div>' % (i, o)
        for i, o in enumerate(options))
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff}
#__UID__ .mc-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .mc-q{color:#444;font-size:13.5px;margin-bottom:12px;line-height:1.55}
#__UID__ .mc-opt{display:flex;align-items:flex-start;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;line-height:1.5;transition:.12s}
#__UID__ .mc-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .mc-dot{width:16px;height:16px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto;margin-top:2px}
#__UID__ .mc-opt code{background:#f3f0ff;border-radius:5px;padding:1px 5px;font-size:12.5px}
#__UID__ .mc-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .mc-opt.sel .mc-dot{background:#764ba2;border-color:#764ba2}
#__UID__ .mc-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .mc-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .mc-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .mc-rev{font-size:13px;color:#2c2350;margin-top:10px;min-height:18px;line-height:1.6}
</style>
<div id="__UID__">
  <div class="mc-head">__TITLE__</div>
  <div class="mc-q">__Q__</div>
  __ROWS__
  <button class="mc-btn">Check my answer</button>
  <div class="mc-rev"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const opts=root.querySelectorAll(".mc-opt"); let sel=null;
  opts.forEach(o=>o.addEventListener("click",()=>{
    sel=+o.dataset.i; opts.forEach(x=>x.classList.remove("sel","ok","no")); o.classList.add("sel");
    root.querySelector(".mc-rev").textContent="";
  }));
  root.querySelector(".mc-btn").addEventListener("click",()=>{
    if(sel===null){root.querySelector(".mc-rev").textContent="Pick an option first!";return;}
    opts.forEach(o=>{const i=+o.dataset.i; o.classList.remove("sel");
      if(i===D.ans)o.classList.add("ok"); else if(i===sel)o.classList.add("no");});
    root.querySelector(".mc-rev").innerHTML=(sel===D.ans?"✅ Correct. ":"❌ Not quite. ")+D.reveal;
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__Q__", question).replace("__ROWS__", rows)
            .replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


def _tf_render(title, statements,
               prompt="Click every statement you think is TRUE, then check."):
    items = [{"t": t, "ok": bool(v)} for t, v in statements]
    uid = "tf_" + str(abs(hash((title, tuple(t for t, _ in statements)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff}
#__UID__ .tf-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .tf-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .tf-opt{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:9px 12px;margin-bottom:7px;cursor:pointer;font-size:13.5px}
#__UID__ .tf-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .tf-box{width:16px;height:16px;border-radius:4px;border:2px solid #c2c7da;flex:0 0 auto}
#__UID__ .tf-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .tf-opt.sel .tf-box{background:#764ba2;border-color:#764ba2}
#__UID__ .tf-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .tf-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .tf-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .tf-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="tf-head">__TITLE__</div>
  <div class="tf-sub">__PROMPT__</div>
  <div class="tf-list"></div>
  <button class="tf-btn">Check</button>
  <div class="tf-status"></div>
</div>
<script>
(function(){
  let DATA=__DATA__.slice();
  for(let i=DATA.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[DATA[i],DATA[j]]=[DATA[j],DATA[i]];}
  const root=document.getElementById("__UID__"), list=root.querySelector(".tf-list");
  DATA.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="tf-opt"; row.dataset.i=i;
    row.innerHTML='<span class="tf-box"></span>'+d.t;
    row.addEventListener("click",()=>{row.classList.remove("ok","no");row.classList.toggle("sel");});
    list.appendChild(row);
  });
  root.querySelector(".tf-btn").addEventListener("click",()=>{
    let right=0; const rows=list.querySelectorAll(".tf-opt");
    rows.forEach(r=>{
      const d=DATA[+r.dataset.i], picked=r.classList.contains("sel");
      r.classList.remove("ok","no");
      if(picked===d.ok)right++; else r.classList.add("no");
      if(d.ok)r.classList.add("ok");
    });
    root.querySelector(".tf-status").textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉":" — green = actually true.");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__DATA__", _json.dumps(items)))
    display(HTML(html))


# ===========================================================================
#  Static illustrations
# ===========================================================================
def show_source(*objs):
    """Pretty-print the source of one or more imported objects **with syntax
    highlighting**, the way an editor would show it — far easier to read than a
    plain `print(inspect.getsource(...))`. Falls back to plain text if the rich
    renderer isn't available."""
    import inspect
    src = "\n\n".join(inspect.getsource(o) for o in objs)
    try:
        from IPython.display import Code, display
        display(Code(src, language="python"))
    except Exception:
        print(src)


def _card(inner, maxw=820):
    display(HTML(
        '<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid '
        '#e6e8ee;border-radius:14px;padding:18px;max-width:%dpx;background:#fff;'
        'line-height:1.55">%s</div>' % (maxw, inner)))


def _pipe_svg(segments, leak_on=None):
    """Render a horizontal pipe: container A (full) -> segments of varying width
    (capacity) -> container B. `segments` is a list of (label, width_frac, sub).
    A thin segment = a bottleneck; `leak_on` marks a segment that overflows."""
    seg_html = ""
    for i, (label, w, sub) in enumerate(segments):
        h = int(18 + 64 * w)                      # taller pipe = more capacity
        narrow = w <= 0.30
        fill = "#cfe0ff" if not narrow else "#ffd9c2"
        edge = "#6f7bf0" if not narrow else "#dd8452"
        leak = ""
        if leak_on == i:
            leak = ('<div style="position:absolute;top:-12px;left:50%;transform:translateX(-50%);'
                    'font-size:14px">💧💧</div>')
        tag = ('<div style="font-size:10.5px;font-weight:700;color:%s;margin-top:4px">%s</div>'
               % ("#b4541f" if narrow else "#3b3f8f", "⚠ bottleneck" if narrow else ""))
        seg_html += (
            '<div style="display:flex;flex-direction:column;align-items:center;justify-content:flex-end">'
            '<div style="position:relative;width:74px;height:%dpx;background:%s;border:2px solid %s;'
            'border-radius:6px">%s</div>'
            '<div style="font-size:11.5px;font-weight:700;color:#2b2d6b;margin-top:6px">%s</div>'
            '<div style="font-size:10.5px;color:#777">%s</div>%s</div>'
            '<div style="align-self:center;font-size:18px;color:#9aa">▶</div>'
            % (h, fill, edge, leak, label, sub, tag))
    return (
        '<div style="display:flex;align-items:flex-end;gap:6px;flex-wrap:nowrap;overflow-x:auto;padding:8px 0">'
        # container A — full
        '<div style="display:flex;flex-direction:column;align-items:center">'
        '<div style="width:46px;height:84px;border:2px solid #6f7bf0;border-radius:6px;'
        'background:linear-gradient(#cfe0ff,#cfe0ff);position:relative"></div>'
        '<div style="font-size:11px;font-weight:700;color:#2b2d6b;margin-top:6px">A · full</div></div>'
        '<div style="align-self:center;font-size:18px;color:#9aa">▶</div>'
        + seg_html +
        # container B — empty
        '<div style="display:flex;flex-direction:column;align-items:center">'
        '<div style="width:46px;height:84px;border:2px solid #b9c0d8;border-radius:6px;background:#fff"></div>'
        '<div style="font-size:11px;font-weight:700;color:#777;margin-top:6px">B · empty</div></div>'
        '</div>')


def water_pipe_analogy():
    """Plain plumbing first: move all the water from A to B through a pipe whose
    narrowest segment caps the flow, and whose segments can overflow."""
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:8px">'
        '🚰 First, just plumbing — no ML yet</div>'
        '<p style="font-size:13.5px;color:#444;margin:0 0 6px">'
        'You have a tank <b>A</b> full of water and an empty tank <b>B</b>, joined by a pipe. '
        'The goal is simple: <b>move all the water from A to B as fast as possible.</b> '
        'The pipe is built from segments of different widths:</p>'
        + _pipe_svg([
            ("segment 1", 0.85, "wide"),
            ("segment 2", 0.22, "narrow"),
            ("segment 3", 0.70, "wide"),
        ], leak_on=None) +
        '<p style="font-size:13px;color:#444;margin:10px 0 0">'
        'Two facts fall out of the picture:<br>'
        '• <b>The narrowest segment sets the speed.</b> No matter how wide the others are, the '
        'water can only get through as fast as the thin part lets it. Widening a wide segment '
        'changes nothing — only the <b>bottleneck</b> matters.<br>'
        '• <b>A segment can overflow.</b> Push water in faster than a segment can pass it on and '
        'it backs up and spills over the top.</p>')


def water_pipe_ml():
    """Now ground the same pipe in a training step."""
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:8px">'
        '🔌 Now ground it: one epoch <i>is</i> that pipe</div>'
        '<p style="font-size:13.5px;color:#444;margin:0 0 6px">'
        'The water is your <b>training data</b>; pushing all of it from A to B once is <b>one '
        'epoch</b>. The pipe segments are the stages every batch flows through:</p>'
        + _pipe_svg([
            ("dataloader", 0.22, "load + preprocess"),
            ("CPU→GPU", 0.55, "transfer cost"),
            ("GPU compute", 0.90, "fwd+bwd+update"),
        ], leak_on=2) +
        '<p style="font-size:13px;color:#444;margin:10px 0 0">'
        '• The <b>bottleneck</b> caps throughput — here a slow <b>dataloader</b> starves a fast '
        'GPU (a faster GPU wouldn\'t help at all).<br>'
        '• An <b>overflow</b> is the <b>GPU running out of VRAM</b> — an out-of-memory crash. '
        '(You\'ll see much more on VRAM in a future large-scale-training notebook.)</p>')


def cpu_gpu_split():
    """CPU×RAM vs GPU×VRAM division of labour."""
    _card(r'''
    <div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:4px">
      🧠 Two worlds: CPU × RAM and GPU × VRAM</div>
    <p style="font-size:13px;color:#444;margin:0 0 10px">
      Each processor has its <b>own</b> memory, joined by a bus. The CPU works out of <b>RAM</b>
      (big, cheap — treat it as ~infinite); the GPU works out of <b>VRAM</b> (small and precious).
      Notice the size difference — that's the whole reason VRAM is the thing we fight over.</p>
    <div style="display:flex;align-items:center;justify-content:center;gap:18px;margin:4px 0 18px">
      <div style="text-align:center">
        <div style="width:240px;height:140px;border:2px solid #6f7bf0;border-radius:12px;
                    background:#eef3ff;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;gap:8px">
          <div style="font-weight:800;color:#2b2d6b;font-size:15px">RAM</div>
          <div style="font-size:11px;color:#555">big &amp; cheap — treat as ~infinite</div>
          <div style="width:60px;height:40px;border:2px solid #2b2d6b;border-radius:7px;background:#fff;
                      display:flex;align-items:center;justify-content:center;font-size:12px;
                      font-weight:800;color:#2b2d6b">CPU</div>
        </div>
      </div>
      <div style="text-align:center;color:#9aa">
        <div style="font-size:24px;line-height:1">⇄</div>
        <div style="font-size:10px;color:#777">bus<br>(copy cost)</div>
      </div>
      <div style="text-align:center">
        <div style="width:130px;height:92px;border:2px solid #dd8452;border-radius:12px;
                    background:#fff4ec;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;gap:6px">
          <div style="font-weight:800;color:#b4541f;font-size:14px">VRAM</div>
          <div style="font-size:10.5px;color:#555">small &amp; precious</div>
          <div style="width:54px;height:34px;border:2px solid #b4541f;border-radius:7px;background:#fff;
                      display:flex;align-items:center;justify-content:center;font-size:12px;
                      font-weight:800;color:#b4541f">GPU</div>
        </div>
      </div>
    </div>
    <table style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="background:#f6f7fb">
        <th style="text-align:left;padding:8px 12px;border:1px solid #e2e5ef"></th>
        <th style="text-align:left;padding:8px 12px;border:1px solid #e2e5ef">CPU × RAM</th>
        <th style="text-align:left;padding:8px 12px;border:1px solid #e2e5ef">GPU × VRAM</th></tr>
      <tr><td style="padding:8px 12px;border:1px solid #e2e5ef;font-weight:700">Memory</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">huge, swappable to disk → treat as ~infinite</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">limited &amp; precious → the thing we optimize</td></tr>
      <tr style="background:#fbfbfe"><td style="padding:8px 12px;border:1px solid #e2e5ef;font-weight:700">Good at</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">branchy, sequential logic; I/O; orchestration</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">massively parallel <i>batched</i> math — <b>matrix multiplications</b> (<i>matmuls</i> for short)</td></tr>
      <tr><td style="padding:8px 12px;border:1px solid #e2e5ef;font-weight:700">In the pipe</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">feeds the water in (loads/preprocesses)</td>
        <td style="padding:8px 12px;border:1px solid #e2e5ef">the wide, fast segment — where training math runs</td></tr>
    </table>
    <p style="font-size:13px;color:#444;margin:12px 0 0">
      Training is mostly big batched matmuls — exactly the GPU's strength. Doing that math on
      the CPU narrows the pipe enormously, which is why we don't train on CPU. We talk about a
      GPU's <b>FLOP/s</b> (operations on the data per second); going from FLOP/s to the pipe's
      real <b>bytes/second</b> is not a clean conversion — it depends on the op and the data.</p>''')


def _bits_row(name, sign, exp, mant, note):
    """One float format as a row of coloured bit-boxes (1 sign / exp / mantissa).
    Box widths are proportional to the bit counts, so 16- vs 32-bit is visible."""
    unit = 9   # px per bit
    def block(n, color, label):
        if n == 0:
            return ""
        return ('<div style="width:%dpx;height:26px;background:%s;border:1px solid #fff;'
                'display:flex;align-items:center;justify-content:center;font-size:10px;'
                'font-weight:700;color:#fff;white-space:nowrap">%s</div>'
                % (n * unit, color, label))
    bar = (block(sign, "#555", "S")
           + block(exp, "#dd8452", "%d exp" % exp)
           + block(mant, "#6f7bf0", "%d mantissa" % mant))
    total = sign + exp + mant
    return (
        '<div style="display:flex;align-items:center;gap:12px;margin:7px 0">'
        '<div style="width:46px;font-weight:800;font-size:13px;color:#2b2d6b">%s</div>'
        '<div style="display:flex;border-radius:5px;overflow:hidden">%s</div>'
        '<div style="font-size:11.5px;color:#777">%d bits — %s</div></div>'
        % (name, bar, total, note))


def _num_line(label, sub, ticks, reach_lo, reach_hi, color,
              full_lo=-8.0, full_hi=8.0, target=3.3, width=540):
    """One schematic 1-D number line on a shared scale [full_lo, full_hi].

    `ticks` are the representable values (tick density = precision); the coloured
    bar from `reach_lo`..`reach_hi` is the format's range. A ▼ marks a target
    value and a dashed drop shows where it lands — snapped to the nearest tick if
    in range, or falling off the end if out of range.
    """
    def x(v):
        return (max(full_lo, min(full_hi, v)) - full_lo) / (full_hi - full_lo) * width
    in_range = reach_lo <= target <= reach_hi
    nearest = min(ticks, key=lambda t: abs(t - target)) if in_range else None

    parts = ['<div style="position:relative;height:46px;width:%dpx;margin:2px 0">' % width]
    # baseline + reachable bar
    parts.append('<div style="position:absolute;left:0;right:0;top:24px;height:2px;background:#dde0ea"></div>')
    parts.append('<div style="position:absolute;left:%dpx;width:%dpx;top:23px;height:4px;'
                 'background:%s;border-radius:2px;opacity:.85"></div>'
                 % (x(reach_lo), x(reach_hi) - x(reach_lo), color))
    # range-end caps
    for v in (reach_lo, reach_hi):
        parts.append('<div style="position:absolute;left:%dpx;top:17px;width:2px;height:16px;'
                     'background:%s"></div>' % (x(v) - 1, color))
    # ticks
    for v in ticks:
        parts.append('<div style="position:absolute;left:%dpx;top:20px;width:1px;height:10px;'
                     'background:%s;opacity:.7"></div>' % (x(v), color))
    # target marker
    parts.append('<div style="position:absolute;left:%dpx;top:-2px;transform:translateX(-50%%);'
                 'font-size:12px;color:#222">▼</div>' % x(target))
    if in_range:
        parts.append('<div style="position:absolute;left:%dpx;top:8px;width:1px;height:18px;'
                     'background:#222;opacity:.4"></div>' % x(nearest))
        verdict = ('<span style="color:#2e7d4f">→ rounds to nearest tick (%g)</span>' % nearest)
    else:
        parts.append('<div style="position:absolute;left:%dpx;top:8px;font-size:12px;color:#c0392b">'
                     '✗</div>' % (x(reach_hi) + 6))
        verdict = '<span style="color:#c0392b">→ off the end: cannot represent it</span>'
    parts.append('</div>')
    track = "".join(parts)
    return ('<div style="display:flex;align-items:center;gap:14px;margin:4px 0 2px">'
            '<div style="width:48px;font-weight:800;font-size:13px;color:%s">%s</div>'
            '<div>%s<div style="font-size:11px;color:#777;margin-top:-4px">%s · %s</div></div></div>'
            % (color, label, track, sub, verdict))


def fp32_vs_fp16():
    """Bit-layout comparison of fp32 / fp16 / bf16, plus a schematic number line
    that shows the range↔precision trade-off (and where each format places the
    same target value)."""
    blue, orange, green = "#6f7bf0", "#dd8452", "#2e9e7a"
    fp32_ticks = [(-8 + 0.5 * i) for i in range(33)]       # wide + dense
    bf16_ticks = list(range(-8, 9))                        # wide + coarse
    fp16_ticks = [(-2 + 0.25 * i) for i in range(17)]      # narrow + dense
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:12px">'
        '🔢 How a weight is stored: fp32 vs fp16 vs bf16</div>'
        '<div style="display:flex;gap:14px;font-size:11px;color:#555;margin-bottom:6px">'
        '<span>🟪 <b>mantissa</b> = precision</span><span>🟧 <b>exponent</b> = range</span>'
        '<span>⬛ sign</span></div>'
        + _bits_row("fp32", 1, 8, 23, "wide range, fine precision")
        + _bits_row("fp16", 1, 5, 10, "small range, good precision")
        + _bits_row("bf16", 1, 8, 7, "fp32's range, coarse precision")
        + '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin:12px 0 14px;'
        'font-size:12.5px;color:#333;line-height:1.6">'
        'Two knobs, and the 16-bit formats spend their bits differently:<br>'
        '• more <span style="color:#b4541f"><b>exponent</b></span> bits → bigger <b>range</b> '
        '(you can reach far bigger and far tinier numbers);<br>'
        '• more <span style="color:#5b63c4"><b>mantissa</b></span> bits → finer <b>precision</b> '
        '(you can tell <code>1.50</code> apart from <code>1.5001</code>).</div>'
        + '<div style="font-size:12.5px;color:#444;margin-bottom:2px">'
        'The same idea on a number line — the dots are the values each format can store, '
        'and the <b>▼</b> is one target value (<code>3.3</code>) we try to store:</div>'
        + _num_line("fp32", "wide range &amp; fine ticks", fp32_ticks, -8, 8, blue)
        + _num_line("fp16", "fine ticks, but narrow range", fp16_ticks, -2, 2, orange)
        + _num_line("bf16", "wide range, but coarse ticks", bf16_ticks, -8, 8, green)
        + '<p style="font-size:13px;color:#444;margin:12px 0 0">'
        'That picture is the whole trade-off. <b>fp16</b> keeps fine ticks but a <b>narrow range</b> '
        '— so very tiny gradients fall off the low end and underflow to 0 (vanishing gradients). '
        '<b>bf16</b> keeps fp32\'s <b>range</b> (same 8 exponent bits) and pays with coarser ticks '
        '(precision) instead: nothing underflows, values just round to the nearest tick — usually '
        'the safer trade for training.</p>')


def over_underfit_demo():
    """A from-scratch, non-churn illustration of under/good/over-fitting: fit
    polynomials of growing degree to noisy 1D points so the three regimes are
    visible before we go anywhere near the model."""
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 14)
    true = np.sin(2 * np.pi * x)                      # the real pattern
    y = true + rng.normal(0, 0.18, size=x.shape)      # noisy observations
    xs = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [(1, "Underfitting", "too simple — misses the pattern (high train & val error)"),
              (3, "Good fit", "captures the trend, ignores the noise"),
              (12, "Overfitting", "memorises the noise — low train error, high val error")]
    for ax, (deg, title, sub) in zip(axes, panels):
        coef = np.polyfit(x, y, deg)
        ax.scatter(x, y, color="#2b2d6b", s=28, zorder=3, label="training data")
        ax.plot(xs, np.sin(2 * np.pi * xs), "--", color="#999", lw=1.5, label="true pattern")
        ax.plot(xs, np.polyval(coef, xs), color="#c44e52", lw=2, label="model")
        ax.set_ylim(-1.8, 1.8); ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel(sub, fontsize=9.5, color="#555")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Under- vs over-fitting, on a toy 1D dataset (not churn)",
                 fontweight="bold")
    plt.tight_layout(); plt.show()


def autograd_graph():
    """High-level view of the autograd graph kept for backprop."""
    _card(r'''
    <div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:8px">
      🕸️ Why inference can hog memory: the autograd graph</div>
    <p style="font-size:13.5px;color:#444;margin:0 0 10px">
      During the forward pass PyTorch quietly records every operation and <b>keeps the
      intermediate tensors</b> — it needs them to compute gradients in the backward pass.</p>
    <pre style="font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f6f7fb;
      border-radius:8px;padding:12px;margin:0;color:#333">
  x ─▶ [matmul] ─▶ h1 ─▶ [relu] ─▶ h2 ─▶ [matmul] ─▶ out
        keep x,W      keep h1       keep h2,W        (all saved
        ───────────── saved for backward ──────────  for grads)</pre>
    <p style="font-size:13px;color:#444;margin:10px 0 0">
      At <b>inference</b> we never call <code>.backward()</code>, so storing all that is pure
      waste. <code>torch.no_grad()</code> / <code>torch.inference_mode()</code> tell PyTorch to
      <b>stop recording</b> — the intermediates are freed immediately and the memory footprint
      drops, so you can serve far more requests at once.</p>''')


# ===========================================================================
#  einops visualisations  (trace where each element goes)
# ===========================================================================
# soft fill / strong edge pairs, indexed by "group" (channel, patch, batch…)
_EIN_BG = ["#dbe4ff", "#ffe8cc", "#d8f5e3", "#f3d9fa", "#fff3bf", "#ffd8d8"]
_EIN_FG = ["#3b5bdb", "#e8590c", "#2f9e44", "#9c36b5", "#e67700", "#c92a2a"]


def _ein_cell(txt, g=None, w=30, h=26):
    """One labelled tensor element; `g` picks a group colour (None = plain)."""
    if g is None:
        bg, fg, bd = "#f4f5f9", "#555", "#e2e5ef"
    else:
        bg, fg, bd = _EIN_BG[g % len(_EIN_BG)], _EIN_FG[g % len(_EIN_FG)], "rgba(0,0,0,.08)"
    return ('<div style="width:%dpx;height:%dpx;display:flex;align-items:center;justify-content:'
            'center;font-size:11px;font-weight:600;border-radius:5px;background:%s;color:%s;'
            'border:1px solid %s">%s</div>' % (w, h, bg, fg, bd, txt))


def _ein_grid(rows, gap=3):
    """`rows` = list of rows; each row = list of pre-rendered cell HTML strings."""
    body = "".join('<div style="display:flex;gap:%dpx">%s</div>' % (gap, "".join(r))
                   for r in rows)
    return '<div style="display:flex;flex-direction:column;gap:%dpx">%s</div>' % (gap, body)


def _ein_block(title, inner):
    return ('<div style="display:flex;flex-direction:column;align-items:center;gap:6px">'
            '<div style="font-size:11px;font-weight:700;color:#555">%s</div>%s</div>'
            % (title, inner))


def _ein_arrow(pattern):
    return ('<div style="display:flex;flex-direction:column;align-items:center;'
            'justify-content:center;color:#9aa;padding:0 4px">'
            '<div style="font-size:22px;line-height:1">▶</div>'
            '<code style="font-size:10.5px;color:#6b5bd0;background:#f1edff;border-radius:5px;'
            'padding:1px 5px;margin-top:3px;white-space:nowrap">%s</code></div>' % pattern)


def _ein_flow(*blocks_and_arrows, note=""):
    inner = ('<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
             'justify-content:center;padding:4px 0">' + "".join(blocks_and_arrows) + '</div>')
    if note:
        inner += ('<p style="font-size:12.5px;color:#444;margin:12px 0 0;line-height:1.55">%s</p>'
                  % note)
    return inner


def channels_intro_viz():
    """What 'channel' really means: one small RGB image (4×4) split into its three
    colour planes. Intensities 0..9 (0 = none of that colour, 9 = full)."""
    R = [[9, 9, 2, 1], [9, 7, 1, 1], [1, 1, 7, 9], [0, 1, 9, 9]]
    G = [[1, 2, 2, 1], [2, 3, 4, 3], [3, 4, 3, 2], [2, 3, 2, 1]]
    B = [[1, 1, 8, 9], [1, 2, 9, 8], [8, 9, 2, 1], [9, 8, 1, 1]]

    def comp_cell(r, g, b):
        return ('<div style="width:30px;height:30px;border-radius:5px;border:1px solid '
                'rgba(0,0,0,.08);background:rgb(%d,%d,%d)"></div>'
                % (int(r / 9 * 255), int(g / 9 * 255), int(b / 9 * 255)))

    def grid(rows_html):
        return '<div style="display:flex;flex-direction:column;gap:3px">%s</div>' % rows_html

    composite = grid("".join(
        '<div style="display:flex;gap:3px">%s</div>'
        % "".join(comp_cell(R[i][j], G[i][j], B[i][j]) for j in range(4)) for i in range(4)))

    def plane(M, hue):
        def cell(v):
            inten = int(v / 9 * 255)
            if hue == "r":
                bg = "rgb(255,%d,%d)" % (255 - inten, 255 - inten)
            elif hue == "g":
                bg = "rgb(%d,255,%d)" % (255 - inten, 255 - inten)
            else:
                bg = "rgb(%d,%d,255)" % (255 - inten, 255 - inten)
            return ('<div style="width:30px;height:30px;display:flex;align-items:center;'
                    'justify-content:center;font-size:11px;font-weight:600;color:#333;'
                    'border-radius:5px;border:1px solid rgba(0,0,0,.08);background:%s">%d</div>'
                    % (bg, v))
        return grid("".join('<div style="display:flex;gap:3px">%s</div>'
                            % "".join(cell(M[i][j]) for j in range(4)) for i in range(4)))

    arrow = ('<div style="display:flex;align-items:center;color:#9aa;font-size:22px;'
             'padding:0 6px">▶</div>')
    planes = (_ein_block("🔴 red channel", plane(R, "r"))
              + _ein_block("🟢 green channel", plane(G, "g"))
              + _ein_block("🔵 blue channel", plane(B, "b")))
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🖼️ What a "channel" is: a colour image = 3 stacked grids of numbers</div>'
        '<p style="font-size:12.5px;color:#444;margin:0 0 10px;line-height:1.55">'
        'A computer doesn\'t see colours — it sees <b>numbers</b>. A colour image is three '
        'grids stacked on top of each other: how much <b>red</b>, how much <b>green</b>, how much '
        '<b>blue</b> at each pixel. Those three grids are the <b>channels</b>.</p>'
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:center">'
        + _ein_block("the image (4×4 pixels)", composite) + arrow
        + '<div style="display:flex;gap:10px">' + planes + '</div></div>'
        + '<p style="font-size:12.5px;color:#444;margin:12px 0 0;line-height:1.55">'
        'So one image is <code>(channel, height, width)</code> = <code>(3, 4, 4)</code> here. '
        'Stack several images to process them together and you prepend a <b>batch</b> axis: '
        '<code>(batch, channel, height, width)</code> — the <code>b c h w</code> you keep seeing. '
        '<b>height</b> and <b>width</b> are just the pixel rows and columns.</p>')


def einops_flatten_viz():
    """`b c h w -> b (c h w)`: three 2×2 channel planes flatten into one row,
    colour-grouped by channel so the merge order is visible."""
    planes = ""
    for c in range(3):
        plane = _ein_grid([[_ein_cell(str(c * 4 + 0), c), _ein_cell(str(c * 4 + 1), c)],
                           [_ein_cell(str(c * 4 + 2), c), _ein_cell(str(c * 4 + 3), c)]])
        planes += _ein_block("channel %d" % c, plane)
    before = ('<div style="display:flex;gap:10px">%s</div>' % planes)
    after = _ein_grid([[_ein_cell(str(i), i // 4) for i in range(12)]])
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔎 <code>b c h w -&gt; b (c h w)</code> — merge channels &amp; pixels into one row</div>'
        + _ein_flow(
            _ein_block("(b=1) c=3, h=2, w=2", before),
            _ein_arrow("b (c h w)"),
            _ein_block("(b=1) features=12", after),
            note="The merged row is filled like an <b>odometer</b>: the right-most name in "
            "<code>(c h w)</code> ticks fastest. So we lay down all of channel&nbsp;0's pixels "
            "first (<b>0,1,2,3</b>), then channel&nbsp;1 (<b>4,5,6,7</b>), then channel&nbsp;2 — "
            "we only step to the next channel once the current one is finished. Same colour = same "
            "channel, so you can see nothing gets shuffled."))


def einops_permute_viz():
    """`b h w c -> b c h w`: pixel-major (each pixel holds its 3 channels) becomes
    channel-major (3 planes). Colour = channel."""
    # before: 2x2 pixels, each pixel a little strip of its 3 channel values
    def pixel(h, w):
        base = (h * 2 + w) * 3
        strip = _ein_grid([[_ein_cell(str(base + c), c, w=22, h=20) for c in range(3)]])
        return _ein_block("px(%d,%d)" % (h, w), strip)
    before = ('<div style="display:flex;flex-direction:column;gap:6px">'
              '<div style="display:flex;gap:6px">%s%s</div>'
              '<div style="display:flex;gap:6px">%s%s</div></div>'
              % (pixel(0, 0), pixel(0, 1), pixel(1, 0), pixel(1, 1)))
    # after: 3 planes, plane c at (h,w) = (h*2+w)*3 + c
    planes = ""
    for c in range(3):
        plane = _ein_grid([[_ein_cell(str((h * 2 + w) * 3 + c), c) for w in range(2)]
                           for h in range(2)])
        planes += _ein_block("channel %d" % c, plane)
    after = ('<div style="display:flex;gap:10px">%s</div>' % planes)
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔎 <code>b h w c -&gt; b c h w</code> — regroup by channel (a permute)</div>'
        + _ein_flow(
            _ein_block("pixel-major (each pixel: 3 channels)", before),
            _ein_arrow("b h w c -> b c h w"),
            _ein_block("channel-major (3 planes)", after),
            note="No number is created or destroyed — only <b>regrouped</b>. Before, the three "
            "channels of one pixel sit together; after, each channel is gathered into its own "
            "plane. Following one colour shows exactly where its values travel."))


def einops_reduce_viz():
    """`b c h w -> b c` with mean: each 2×2 channel plane collapses to one number."""
    blocks = ""
    means = [1.5, 5.5, 9.5]
    for c in range(3):
        plane = _ein_grid([[_ein_cell(str(c * 4 + 0), c), _ein_cell(str(c * 4 + 1), c)],
                           [_ein_cell(str(c * 4 + 2), c), _ein_cell(str(c * 4 + 3), c)]])
        out = _ein_cell(("%g" % means[c]), c, w=44, h=44)
        blocks += ('<div style="display:flex;align-items:center;gap:8px">'
                   + _ein_block("channel %d" % c, plane)
                   + '<div style="color:#9aa;font-size:13px">mean▶</div>'
                   + _ein_block("", out) + '</div>')
    inner = ('<div style="display:flex;gap:18px;flex-wrap:wrap;justify-content:center">%s</div>'
             % blocks)
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '🔎 <code>b c h w -&gt; b c</code>, "mean" — average the spatial axes away</div>'
        + inner
        + '<p style="font-size:12.5px;color:#444;margin:12px 0 0;line-height:1.55">'
        'The names that <b>disappear</b> on the right (<code>h</code>, <code>w</code>) are the ones '
        'being averaged: every 2×2 plane collapses to a <b>single per-channel number</b>. The '
        'pattern <i>is</i> the formula — that\'s the whole appeal of <code>reduce</code>.</p>')


def _ein_label(txt, color="#777"):
    return ('<div style="font-size:10.5px;font-weight:700;color:%s;text-align:center;'
            'margin:2px 0">%s</div>' % (color, txt))


def einops_patches_viz(show_pattern=True):
    """The patchify move built up in stages, on a 4×6 single-channel image cut
    into a 2×3 grid of 2×2 patches (so h≠w are visible), plus a final note on
    where the channel goes.

    ``show_pattern=True``  → also prints the einops line ``(h p1)(w p2) ->
    (h w)(p1 p2)`` and labels the axes with their einops names. Use this *after*
    the student has written the pattern.
    ``show_pattern=False`` → the same picture as a **blueprint of the goal** with
    NO einops syntax shown, so it can be used to *pose* the challenge without
    spoiling the answer."""
    H, W, P1, P2 = 2, 3, 2, 2          # 2 patch-rows × 3 patch-cols, each patch 2×2
    rows, cols = H * P1, W * P2        # 4 × 6 image
    def patch_of(r, c):
        return (r // P1) * W + (c // P2)
    # ---- Stage 1: the image, coloured into patches, with axis labels ----
    img = _ein_grid([[_ein_cell(str(r * cols + c), patch_of(r, c))
                      for c in range(cols)] for r in range(rows)])
    h_lbl = "height = h·p1 (h=2 patches × p1=2)" if show_pattern else "height = 2 patches × 2 pixels"
    w_lbl = "width = w·p2  (w=3 patches × p2=2)" if show_pattern else "width = 3 patches × 2 pixels"
    stage1 = (
        '<div style="display:flex;gap:8px;align-items:center">'
        '<div style="writing-mode:vertical-rl;transform:rotate(180deg);font-size:10.5px;'
        'font-weight:700;color:#777">' + h_lbl + '</div>'
        '<div>' + _ein_label(w_lbl) + img + '</div></div>')
    # ---- Stage 2: each patch -> its own row, pixels flattened ----
    after = _ein_grid([[
        _ein_cell(str(((p // W) * P1 + pr) * cols + ((p % W) * P2 + pc)), p)
        for pr in range(P1) for pc in range(P2)] for p in range(H * W)])
    px_lbl  = "(p1 p2) = 4 pixels in the patch →" if show_pattern else "4 pixels in each patch →"
    row_lbl = "↑ (h w) = 6 patches, one per row" if show_pattern else "↑ 6 patches, one per row"
    stage2 = ('<div>' + _ein_label(px_lbl) + after
              + _ein_label(row_lbl, "#555") + '</div>')
    # ---- text that differs depending on whether we reveal the einops names ----
    if show_pattern:
        head = ('🔎 <code>(h p1)(w p2) -&gt; (h w)(p1 p2)</code> — image → one row per patch')
        p1_txt = (
            'Splitting <code>(h p1)</code> reads the height as <b>h=2 patches</b> of <b>p1=2</b> '
            'pixels each; <code>(w p2)</code> reads the width as <b>w=3 patches</b> of <b>p2=2</b> '
            'pixels. That carves the picture into <b>h×w = 6</b> coloured patches.')
        p2_txt = (
            'Merging <code>(h w)</code> in the output just <b>enumerates the patches</b> — 2 rows × '
            '3 cols of patches = <b>6 rows</b>. Merging <code>(p1 p2)</code> flattens each little '
            '2×2 patch into a line of 4 pixels — exactly the "image → one row" move from Example 1.')
        p3_head = '③ And the channel? <code>(c p1 p2)</code>'
        p3_txt = (
            'A real image has 3 colour channels. We want each patch-row to carry <b>all</b> of its '
            'pixels <b>across every channel</b>, so the output groups <code>(c p1 p2)</code>: for '
            'our 2×2 patch that\'s <b>3 × 4 = 12</b> numbers per row instead of 4. Note <code>c</code> '
            'comes <b>first</b>: that keeps each channel\'s pixels <b>contiguous</b> (all red, then '
            'all green, then all blue). A trailing <code>(p1 p2 c)</code> would interleave them '
            '(R,G,B, R,G,B, …) — same numbers, scrambled layout.')
    else:
        head = ('🎯 The goal: turn an image into <b>one row per patch</b> (no einops yet — '
                'this is what you\'ll build)')
        p1_txt = (
            'Read the height as <b>2 patches of 2 pixels</b> each, and the width as <b>3 patches '
            'of 2 pixels</b>. That carves the picture into <b>2 × 3 = 6</b> coloured patches.')
        p2_txt = (
            '<b>Enumerate the patches</b> — 2 rows × 3 cols of patches = <b>6 rows</b> — and '
            'flatten each little 2×2 patch into a line of <b>4 pixels</b>, just like the '
            '"image → one row" move from Example 1.')
        p3_head = '③ And the channel?'
        p3_txt = (
            'A real image has 3 colour channels. We want each patch-row to carry <b>all</b> of its '
            'pixels <b>across every channel</b>, with each channel kept <b>together</b> (all red, '
            'then all green, then all blue — not interleaved): <b>3 × 4 = 12</b> numbers per row '
            'instead of 4.')
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:6px">'
        + head + '</div>'
        # stage 1
        '<div style="font-size:12px;font-weight:700;color:#3b3f8f;margin:2px 0 4px">'
        '① Cut the image into patches (within one channel)</div>'
        + stage1
        + '<p style="font-size:12px;color:#444;margin:8px 0 0;line-height:1.5">' + p1_txt + '</p>'
        # stage 2
        '<div style="font-size:12px;font-weight:700;color:#3b3f8f;margin:14px 0 4px">'
        '② We want each patch on its own row</div>'
        '<div style="display:flex;gap:10px;align-items:center">' + stage2 + '</div>'
        + '<p style="font-size:12px;color:#444;margin:8px 0 0;line-height:1.5">' + p2_txt + '</p>'
        # stage 3: channels
        '<div style="font-size:12px;font-weight:700;color:#3b3f8f;margin:14px 0 4px">'
        + p3_head + '</div>'
        '<p style="font-size:12px;color:#444;margin:0;line-height:1.5">' + p3_txt + '</p>')


def einops_order_viz():
    """`b (n d)` vs `n (b d)` on the same (2,3,4) tensor — same letters, different
    order, different result. The source tensor is shown first so each transform
    reads as 'from this → to that'. Colour = batch, so the mixing is obvious."""
    # x[b,n,d] = b*12 + n*4 + d ; colour by batch b
    src = ""
    for b in range(2):
        grid = _ein_grid([[_ein_cell(str(b * 12 + n * 4 + d), b) for d in range(4)]
                          for n in range(3)])
        src += _ein_block("batch %d (n=3 rows, d=4 cols)" % b, grid)
    source = '<div style="display:flex;gap:12px;justify-content:center">%s</div>' % src
    bnd = _ein_grid([[_ein_cell(str(b * 12 + i), b) for i in range(12)] for b in range(2)])
    nbd = _ein_grid([[_ein_cell(str(b * 12 + n * 4 + d), b) for b in range(2) for d in range(4)]
                     for n in range(3)])
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔎 same letters, different order — <code>b (n d)</code> vs <code>n (b d)</code></div>'
        '<div style="font-size:11.5px;color:#777;margin-bottom:8px">colour = which batch a value '
        'came from (🟦 batch 0, 🟧 batch 1)</div>'
        # the source tensor, shown once
        '<div style="font-size:12px;font-weight:700;color:#3b3f8f;margin:2px 0 4px">'
        'we start from <code>x</code>, shape (b=2, n=3, d=4)</div>'
        + source
        + '<div style="text-align:center;font-size:20px;color:#9aa;margin:8px 0 2px">▼ rearrange ▼</div>'
        # transform 1
        + _ein_flow(_ein_block(
            "x → b (n d) → (2, 12) · each row is ONE whole batch", bnd), note="")
        # transform 2
        + _ein_flow(_ein_block(
            "x → n (b d) → (3, 8) · each row MIXES both batches", nbd),
            note="Both start from the <i>same</i> <code>x</code> above. <code>b (n d)</code> keeps "
            "each batch on its own row — clean. <code>n (b d)</code> puts <code>b</code> "
            "<i>inside</i> the merged axis, so every row now interleaves both batches (🟦 and 🟧 "
            "side by side). A raw <code>.view(...)</code> would give you the second one with no "
            "warning — that's the bug class this whole section is about."))


# ===========================================================================
#  Quiz answer keys  (hidden from the notebook)
# ===========================================================================
_MC_QUIZZES = {
    "pipe_analogy": (
        "🚰 Did the analogy land?",
        "Your dataloader segment runs at 100 MB/s, the CPU→GPU transfer at 5 GB/s, and the "
        "GPU compute at 50 GB/s. You buy a GPU twice as fast. What happens to one epoch's time?",
        ["It roughly halves — the GPU does the heavy compute, so doubling it doubles throughput.",
         "It doubles — a faster GPU pulls data through the whole pipe proportionally faster.",
         "It barely changes — the slow dataloader, not the GPU, caps the throughput.",
         "It can't be predicted from these throughput numbers alone; you'd have to profile."],
        2,
        "Throughput is set by the <b>narrowest</b> segment. The dataloader (100 MB/s) is 500× "
        "slower than the GPU, so a faster GPU just waits longer. Fix the bottleneck first."),
    "why_move_batches": (
        "🚚 You moved the model to the GPU but the batches stayed on the CPU.",
        "You called <code>model.to('cuda')</code> but left each batch on the CPU. What happens?",
        ["A runtime error: an op gets a CPU tensor and a CUDA tensor and refuses to mix them.",
         "It works fine — PyTorch moves the inputs to the right device automatically.",
         "It runs, but silently on the CPU, ignoring the GPU entirely.",
         "The model is quietly moved back to the CPU to match the data."],
        0,
        "Every tensor in an operation must live on the <b>same device</b>. A CPU input meeting "
        "GPU weights raises <code>Expected all tensors to be on the same device</code>. You must "
        "move <b>both</b> the model and every batch with <code>.to(device)</code>."),
    "fp16_speedup": (
        "⚡ When does fp16 actually make training faster?",
        "The team halved the weights to fp16 expecting a 2× speedup. In which hardware setting "
        "does fp16 genuinely run faster?",
        ["Only Setting A — padding each fp16 back to fp32 still runs faster.",
         "Only Setting B — packing two fp16 per unit is the sole real speedup.",
         "Only Setting C — dedicated fp16 units are the only way to gain speed.",
         "Both B and C — the packing setting and the dedicated-fp16-unit setting."],
        3,
        "Speed comes from the <b>hardware</b>, not the dtype label. Padding (A) does the same "
        "fp32 work, so no gain. Packing two fp16 per unit (B) and dedicated faster fp16 units (C) "
        "both do more useful work per cycle — those are where fp16 pays off."),
    "amp_fix": (
        "🩹 Why do bf16 / the grad scaler fix the fp16 problem?",
        "fp16 made gradients underflow to zero. Why do a GradScaler and bf16 both help?",
        ["They make the GPU faster, so there's less wall-clock time for grads to underflow.",
         "The scaler lifts tiny gradients above fp16's underflow floor; bf16 keeps fp32's range.",
         "They raise the learning rate automatically to compensate for the lost precision.",
         "They cast the whole model back to fp32, which removes the speed benefit entirely."],
        1,
        "Two routes to the same goal — keep small gradients representable. The <b>scaler</b> "
        "shifts grads up out of the underflow zone (then unscales before the step); <b>bf16</b> "
        "has fp32's range, so they never underflow. bf16 trades range for precision instead."),
    "dataloader_fix": (
        "🐌 The dataloader is the bottleneck — what do we do?",
        "Profiling points at a <code>very_costly_operation</code> running inside the Dataset's "
        "<code>__getitem__</code>, recomputed for every sample every epoch. Best fix?",
        ["Precompute it once up front, cache the result, then read the cached features.",
         "Run the costly operation on the GPU instead — it's much faster there.",
         "Increase the batch size so <code>__getitem__</code> is called fewer times overall.",
         "Train for fewer epochs so the costly transform runs far less often."],
        0,
        "It's the same transform on the same rows every epoch — pure repeated work. Do it "
        "<b>once</b>, cache, then load the processed features. (Moving an unknown, likely "
        "non-tensor CPU transform to the GPU rarely helps and may not even be possible.)"),
    "inference_mode": (
        "🚀 Inference memory blew up — why, and the one-line fix?",
        "In production the model uses far more memory per request than expected. Most likely "
        "cause and fix?",
        ["The model itself is too big — shrink the architecture to fit in memory.",
         "The input batches aren't on the GPU, so memory thrashes between host and device.",
         "Autograd still saves the backward graph at inference; run under "
         "<code>inference_mode()</code> and call <code>eval()</code>.",
         "The optimizer state is being loaded into memory at inference time."],
        2,
        "Without <code>no_grad</code>/<code>inference_mode</code>, every forward still saves "
        "intermediates for a backward pass that never comes — wasted VRAM. Disable grad tracking "
        "and the footprint drops, so you fit many more concurrent requests."),
}

_TF_QUIZZES = {
    "checkpoint_state": (
        "💾 What carries state we must save to resume training cleanly?",
        [("The model weights (parameters).", True),
         ("The optimizer state — e.g. AdamW keeps running averages of past gradients.", True),
         ("The epoch number / step counter, so we resume at the right place.", True),
         ("The AMP GradScaler's scale factor, when using mixed precision.", True),
         ("The RNG state, so shuffling & dropout continue reproducibly.", True),
         ("Nothing but the weights — everything else is recomputed for free on resume.", False),
         ("The Python source code of the model (it's already on disk).", False),
         ("The training dataset itself, re-saved inside every checkpoint.", False),
         ("The current epoch's running loss average, or the metrics resume wrong.", False)]),
    "fit_diagnosis": (
        "📉 Reading train vs validation loss curves",
        [("Train loss low, val loss high and rising → overfitting.", True),
         ("Train and val loss both high and flat → underfitting (or a bug).", True),
         ("With huge model capacity yet neither loss drops, suspect a code bug, not capacity.", True),
         ("Tracking only the final accuracy tells you whether you over- or under-fit.", False),
         ("A gap between train and val loss always means the data is bad.", False)]),
}


def mc_quiz(key):
    _mc_render(*_MC_QUIZZES[key])


def true_false_quiz(key):
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)
