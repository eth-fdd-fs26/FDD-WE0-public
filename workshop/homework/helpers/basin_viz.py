"""Presentation & quiz helpers for the reactor basin-temperature notebook (HW3).

Same idea as `torch_viz` / `pdm_viz`: all the HTML/CSS illustrations and the quiz
*answer keys* live here, out of the notebook, so the teaching cells stay about the
architecture and the quizzes can't be solved by reading the cell. The notebook does::

    import basin_viz
    basin_viz.pipeline_overview()
    basin_viz.mc_quiz("frozen_blocks")
    basin_viz.show_source(basin_lab.SpectroNet)

Students are told not to read this file.
"""
import json as _json

from IPython.display import HTML, Code, display


# ===========================================================================
#  Source viewer (syntax-highlighted, like torch_viz)
# ===========================================================================
def show_source(*objs):
    """Pretty-print the source of one or more imported objects WITH syntax
    highlighting — far easier to read than `print(inspect.getsource(...))`."""
    import inspect
    src = "\n\n".join(inspect.getsource(o) for o in objs)
    try:
        display(Code(src, language="python"))
    except Exception:
        print(src)


def _card(inner, maxw=860):
    display(HTML(
        '<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid '
        '#e6e8ee;border-radius:14px;padding:18px;max-width:%dpx;background:#fff;'
        'line-height:1.55">%s</div>' % (maxw, inner)))


# ===========================================================================
#  Generic quiz renderers (copied from torch_viz so this module is standalone)
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
#  Quiz answer banks
# ===========================================================================
_MC = {
    "frozen_blocks": (
        "🧊 Reading the plan — what must stay frozen?",
        "The previous engineer marked the thermal-camera <b>vision encoder</b> "
        "“do not touch”. During training, what does that mean we must do?",
        ["Keep training it — more training is always better",
         "Delete it from the model before training",
         "Lower its learning rate to a small value but still update it",
         "Set its parameters' <code>requires_grad=False</code> so the optimizer never updates them"],
        3,
        "Freezing means the optimizer must not touch those weights at all — "
        "<code>requires_grad=False</code> (or wrapping its forward in <code>torch.no_grad()</code>). "
        "A small LR still changes them; deleting it loses the features."),
    "spectronet_extract": (
        "🔌 Re-using SpectroNet",
        "<code>SpectroNet.forward</code> ends in a <code>classifier</code> that outputs class "
        "<i>logits</i> — but we want the <b>features</b> just before it. What's the cleanest way "
        "to get a module that stops at the features?",
        ["Rebuild a module from its children, keeping every one <b>except</b> the final classifier",
         "Copy-paste the conv layers into a brand-new class",
         "Re-train SpectroNet so its output becomes the features",
         "Call <code>spectronet.forward()</code> and ignore the last few numbers of the output"],
        0,
        "The children are registered in order (stem → block1 → pool → classifier), so a module "
        "made from all of them <i>but the last</i> runs everything up to the features. No copying, "
        "no re-training — that's the power of composing <code>nn.Module</code>s. (You'll write the "
        "exact line in the task.)"),
    "skip_shape": (
        "➕ The skip connection",
        "The MLP core adds its <b>input</b> back onto its <b>output</b> (a residual "
        "<code>+ x</code>) before a final ReLU. For that add to even be legal, what must be true?",
        ["Nothing — PyTorch broadcasts any shapes",
         "<code>x</code> must be flattened to 1-D first",
         "The block's output must have the <b>same shape</b> as its input <code>x</code>",
         "The two Linear layers must have a bias"],
        2,
        "A residual add is element-wise, so the inner block must map <code>(B, d) → (B, d)</code> — "
        "input and output dimensions equal. That's why both Linear layers here are "
        "<code>d → d</code>."),
    "conv1d_kernels": (
        "〰️ What is each kernel <i>looking for</i>?",
        "Above, the <b>same</b> signal was run through two length-3 kernels. "
        "<b>Kernel&nbsp;A&nbsp;=&nbsp;[1,&nbsp;0,&nbsp;−1]</b> and "
        "<b>Kernel&nbsp;B&nbsp;=&nbsp;[⅓,&nbsp;⅓,&nbsp;⅓]</b>. From their outputs, which job fits each?",
        ["Both smooth the signal",
         "A averages/smooths the signal · B reacts to where it rises or falls",
         "Both react to slopes",
         "A reacts to where the signal rises or falls (slope) · B averages/smooths it"],
        3,
        "Kernel A subtracts the two ends of its window from each other, so it is ≈0 on flat "
        "stretches and spikes (large ±) wherever the signal <b>changes</b> — a <b>slope / edge</b> "
        "detector, the 1-D "
        "cousin of an image edge filter. Kernel B replaces each point by the average of its "
        "neighbourhood, <b>smoothing</b> out noise. A <code>Conv1d</code> layer <i>learns</i> a "
        "whole bank of such kernels at once."),
    "fusion_dim": (
        "📏 The second intern mistake — a size mismatch",
        "The signal branch outputs <b>32</b> features and the thermal branch outputs <b>32</b>. "
        "We <code>torch.cat</code> them into a <b>single, longer feature vector</b> — the two "
        "32-long vectors joined end-to-end — then feed the MLP core. What input width must the "
        "core expect?",
        ["32", "64", "16", "96"],
        1,
        "<code>cat([s, t], dim=1)</code> stacks the features side by side: 32 + 32 = <b>64</b>. "
        "The intern hard-coded 32, so the very first Linear hits a shape mismatch."),
}

_TF = {
    "pipeline_reading": ("🗺️ Reading the pipeline plan", [
        ("The thermal frames are reshaped BEFORE they enter the vision encoder.", True),
        ("The signal (sensor) branch and the thermal branch are merged by a fusion step.", True),
        ("The vision encoder should be trained along with everything else.", False),
        ("The final block turns the fused features into a single temperature number.", True),
        ("The two input branches can be built in either order; assembly comes last.", True),
        ("The thermal and signal branches share the same encoder weights.", False),
        ("The fused features are turned into one probability per failure class.", False),
        ("The signal branch feeds the thermal branch its features before fusion.", False),
    ]),
    "regression_head": ("🎯 The first intern mistake — the output", [
        ("Predicting a temperature is regression, so the head outputs ONE number per sample.", True),
        ("Without a final layer, the model would output a whole feature vector, not a temperature.", True),
        ("A regression head here should end in a softmax over classes.", False),
        ("MSE loss expects the prediction and the target to have the same shape.", True),
        ("The head needs a sigmoid so the output stays in [0, 1].", False),
        ("The head must output as many numbers as there are input sensors.", False),
    ]),
}


def mc_quiz(key):
    _mc_render(*_MC[key])


def true_false_quiz(key):
    _tf_render(*_TF[key])


# ===========================================================================
#  The pipeline plan — the centrepiece diagram
# ===========================================================================
def _box(title, sub, kind="todo"):
    palette = {
        "input":  ("#eef3ff", "#6f7bf0", "#2b3a8f"),
        "frozen": ("#e9f7ff", "#3aa6c9", "#13627d"),
        "todo":   ("#fff4e6", "#e08a3c", "#9a5410"),
        "done":   ("#edfaf0", "#46b46e", "#1f7a42"),
        "out":    ("#f3edff", "#8a5cd6", "#4a2a8a"),
    }
    bg, edge, fg = palette[kind]
    badge = {"frozen": "❄️ do not touch", "todo": "⚠ to finish",
             "input": "input", "out": "output", "done": ""}[kind]
    badge_html = ('<div style="font-size:9.5px;font-weight:800;color:%s;margin-top:3px;'
                  'text-transform:uppercase;letter-spacing:.3px">%s</div>' % (edge, badge)) if badge else ""
    return ('<div style="background:%s;border:2px solid %s;border-radius:10px;padding:8px 9px;'
            'min-width:96px;flex:0 0 auto;text-align:center">'
            '<div style="font-weight:800;font-size:12px;color:%s">%s</div>'
            '<div style="font-size:10px;color:#555;margin-top:2px">%s</div>%s</div>'
            % (bg, edge, fg, title, sub, badge_html))


def _arrow(vertical=False):
    glyph = "▼" if vertical else "▶"
    return '<div style="color:#9aa;font-size:15px;align-self:center">%s</div>' % glyph


def pipeline_overview():
    """The annotated architecture plan the manager opens first. Reads strictly
    left → right: the two input branches (thermal on top, signal below) run in
    parallel on the LEFT, then both feed a single FUSION → prediction column that
    spans both rows on the RIGHT. Frozen and unfinished blocks are marked."""
    therm = "".join([
        _box("thermal frames", "(B, 3, 16, 16, 1)", "input"), _arrow(),
        _box("reshape (einops)", "permute axes for the encoder", "todo"), _arrow(),
        _box("vision encoder", "→ (B, 32) features", "frozen"), _arrow(),
        _box("thermal feats", "(B, 32)", "done"),
    ])
    sig = "".join([
        _box("sensor series", "(B, 4, 64)", "input"), _arrow(),
        _box("SpectroNet", "extract 1-D-CNN features", "todo"), _arrow(),
        _box("signal feats", "(B, 32)", "done"),
    ])
    row = ('display:flex;gap:6px;align-items:center;flex-wrap:nowrap;'
           'padding:4px 0')

    # the fusion column: a single tall card that spans BOTH branch rows on the right,
    # with its blocks flowing LEFT → RIGHT inside it
    fusion_inner = "".join([
        _box("fusion (concat)", "32 + 32 → 64", "todo"), _arrow(),
        _box("MLP core", "+ skip connection", "todo"), _arrow(),
        _box("regression head", "→ basin °C", "todo"),
    ])
    fusion_card = (
        '<div style="border:2px dashed #b59be0;border-radius:14px;background:#faf7ff;'
        'padding:10px 12px;height:100%%;box-sizing:border-box;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;gap:8px">'
        '<div style="font-size:10.5px;font-weight:800;color:#4a2a8a;text-align:center;'
        'text-transform:uppercase;letter-spacing:.3px">'
        'fusion → prediction &nbsp;<span style="font-weight:600;text-transform:none;'
        'letter-spacing:0;color:#6a5a9a">— both branches meet here</span></div>'
        '<div style="display:flex;flex-direction:row;align-items:center;gap:7px">%s</div>'
        '</div>' % fusion_inner)

    # left column: the two branches stacked, each with its own label
    left = (
        '<div style="flex:0 0 auto;display:flex;flex-direction:column;'
        'justify-content:center;gap:14px">'
        '<div><div style="font-size:11px;font-weight:800;color:#13627d;margin:0 0 2px">'
        'THERMAL BRANCH</div><div style="%s">%s</div></div>'
        '<div><div style="font-size:11px;font-weight:800;color:#9a5410;margin:0 0 2px">'
        'SIGNAL BRANCH</div><div style="%s">%s</div></div>'
        '</div>' % (row, therm, row, sig))

    # the converging connector between the branches and the fusion column
    merge = ('<div style="display:flex;align-items:center;color:#9aa;font-size:22px;'
             'padding:0 2px">▶</div>')

    inner = (
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:4px">'
        '🗺️ The previous engineer\'s pipeline plan</div>'
        '<div style="font-size:12.5px;color:#555;margin-bottom:14px">A confident note reads '
        '<i>“it will work”</i>. Blocks marked <b style="color:#e08a3c">⚠ to finish</b> are yours; '
        'the <b style="color:#3aa6c9">❄️ encoder</b> stays frozen. The two input branches on the '
        '<b>left</b> are <b>independent</b> — do them in any order — then both feed the '
        '<b>fusion column</b> on the right.</div>'
        '<div style="display:flex;align-items:stretch;gap:6px">%s%s%s</div>'
        % (left, merge, fusion_card))
    _card(inner, maxw=980)


def spectronet_diagram():
    """Show SpectroNet's children as boxes, with the classifier crossed out — the
    visual for 'keep everything except the last child'."""
    parts = [
        ("stem", "Conv1d 4→16, ReLU", False),
        ("block1", "Conv1d 16→32, ReLU", False),
        ("pool", "avg over time → (B,32)", False),
        ("classifier", "Linear 32→5  ✂ drop", True),
    ]
    boxes = ""
    for i, (name, sub, drop) in enumerate(parts):
        bg, edge = ("#fdecec", "#e07a7a") if drop else ("#edfaf0", "#46b46e")
        deco = "text-decoration:line-through;opacity:.65" if drop else ""
        boxes += ('<div style="background:%s;border:2px solid %s;border-radius:10px;padding:8px 10px;'
                  'text-align:center;%s"><div style="font-weight:800;font-size:12px">.%s</div>'
                  '<div style="font-size:10px;color:#555">%s</div></div>'
                  % (bg, edge, deco, name, sub))
        if i < len(parts) - 1:
            boxes += _arrow()
    inner = (
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '🔌 SpectroNet, child by child</div>'
        '<div style="font-size:12.5px;color:#555;margin-bottom:12px">Its <code>forward</code> runs '
        'all four children <b>in order</b> and returns class logits. The features we actually want '
        'are the output of <b>pool</b> — everything <i>before</i> the final <code>classifier</code> '
        '(crossed out). So our job is to rebuild a module from these children, stopping one early:</div>'
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">%s</div>'
        '<div style="font-size:12px;color:#1f7a42;margin-top:12px">'
        '→ a new module that runs <b>stem → block1 → pool</b> and outputs <b>(B, 32) features</b>, '
        'never reaching the classifier. <i>(How to build it from the children is your task below.)</i>'
        '</div>' % boxes)
    _card(inner)


# ===========================================================================
#  1-D convolution mini-illustration
# ===========================================================================
def _cell_row(values, edge, bg, fmt="%d"):
    cells = ""
    for v in values:
        txt = ("" if v is None else fmt % v)
        b = "#f6f7fb" if v is None else bg
        e = "#dfe3ee" if v is None else edge
        cells += ('<div style="min-width:38px;height:34px;border:1px solid %s;border-radius:6px;'
                  'display:flex;align-items:center;justify-content:center;font-weight:700;'
                  'font-size:12.5px;background:%s;padding:0 4px">%s</div>' % (e, b, txt))
    return '<div style="display:flex;gap:5px;flex-wrap:wrap">%s</div>' % cells


def conv1d_kernel_gallery():
    """Show ONE signal run through two unnamed kernels (A and B), so students can
    *guess* what each is doing before the quiz reveals it. No labels of function —
    that's the quiz."""
    signal = [1, 1, 2, 1, 6, 6, 5, 6]        # a noisy stretch, then a step up at index 4
    kA = [1, 0, -1]                          # slope / edge detector
    kB = [1 / 3, 1 / 3, 1 / 3]               # smoother

    def slide(kern):
        out = [None]                          # pad so columns line up under the signal
        for i in range(len(signal) - len(kern) + 1):
            out.append(sum(signal[i + j] * kern[j] for j in range(len(kern))))
        out.append(None)
        return out

    inner = (
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '〰️ Two mystery kernels on one signal</div>'
        '<div style="font-size:12.5px;color:#555;margin-bottom:12px">Same sliding mechanic as '
        'above, now with two real length-3 kernels run over the <b>same</b> signal. Study the two '
        'output rows — <b>what is each kernel reacting to?</b> Look at where each output is large '
        'versus near zero. (The quiz below asks you to name them.)</div>'
        '<div style="font-size:11px;font-weight:700;color:#2b3a8f;margin-bottom:4px">'
        'signal &nbsp;(noisy, with a step up partway through)</div>%s'
        '<div style="font-size:11px;font-weight:700;color:#9a5410;margin:12px 0 4px">'
        'Kernel A = [1, 0, −1]</div>%s'
        '<div style="font-size:11px;font-weight:700;color:#1f7a42;margin:12px 0 4px">'
        'Kernel B = [⅓, ⅓, ⅓]</div>%s'
        % (_cell_row(signal, "#6f7bf0", "#eef3ff"),
           _cell_row(slide(kA), "#e08a3c", "#fff4e6", "%+d"),
           _cell_row(slide(kB), "#46b46e", "#edfaf0", "%.1f")))
    _card(inner)


def conv1d_demo():
    """A tiny picture of HOW a length-3 kernel slides along a 1-D signal — the pure
    mechanic (multiply the aligned values, add them into one number, shift right by
    one), shown step by step. Uses a deliberately meaningless kernel so the focus is
    the procedure, not the meaning (that's the gallery + quiz next)."""
    signal = [2, 1, 3, 5, 4, 2]
    kernel = [0, 0, 1]                       # a trivial kernel — just copies the 3rd value
    cw, gap = 34, 5

    def _cell(v, edge, bg):
        return ('<div style="width:%dpx;height:34px;border:1px solid %s;border-radius:6px;'
                'display:flex;align-items:center;justify-content:center;font-weight:700;'
                'font-size:13px;background:%s">%s</div>' % (cw, edge, bg, v))

    sig_row = '<div style="display:flex;gap:%dpx">%s</div>' % (
        gap, "".join(_cell(v, "#b9c0d8", "#eef3ff") for v in signal))

    n = len(signal) - len(kernel) + 1
    steps = ""
    for i in range(n):
        offset = i * (cw + gap)
        krow = "".join(_cell(k, "#e0a64a", "#fff4e6") for k in kernel)
        terms = " + ".join("%d·%d" % (signal[i + j], kernel[j]) for j in range(len(kernel)))
        out = sum(signal[i + j] * kernel[j] for j in range(len(kernel)))
        steps += ('<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                  '<div style="display:flex;gap:%dpx;margin-left:%dpx">%s</div>'
                  '<div style="font-size:12px;color:#555">= %s = '
                  '<b style="color:#1f7a42">%d</b></div></div>' % (gap, offset, krow, terms, out))

    out_cells = "".join(
        _cell(sum(signal[i + j] * kernel[j] for j in range(len(kernel))), "#46b46e", "#edfaf0")
        for i in range(n))
    out_row = '<div style="display:flex;gap:%dpx">%s</div>' % (gap, out_cells)

    inner = (
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '〰️ How a 1-D convolution slides</div>'
        '<div style="font-size:12.5px;color:#555;margin-bottom:10px">A small <b>kernel</b> sits '
        'under the signal. At each position you <b>multiply</b> the aligned values, <b>add</b> them '
        'into a single output number, then <b>shift the kernel one step to the right</b> and repeat. '
        'Here the kernel is the trivial <code>[0, 0, 1]</code> (it just copies the 3rd value) so you '
        'can watch the <i>procedure</i> — the next cell shows kernels that actually detect '
        'something.</div>'
        '<div style="font-size:11px;font-weight:700;color:#2b3a8f;margin-bottom:4px">signal</div>%s'
        '<div style="font-size:11px;font-weight:700;color:#9a5410;margin:12px 0 4px">'
        'kernel [0, 0, 1] sliding right, one step at a time</div>%s'
        '<div style="font-size:11px;font-weight:700;color:#1f7a42;margin:12px 0 4px">'
        'output (one number per position)</div>%s'
        '<div style="font-size:11.5px;color:#777;margin-top:10px">A <code>Conv1d</code> layer learns '
        '<i>many</i> such kernels at once (here 16, then 32), and stacks them as output channels.</div>'
        % (sig_row, steps, out_row))
    _card(inner)


# ===========================================================================
#  Final evaluation plot
# ===========================================================================
def prediction_plot(pred_std, true_std, y_mean, y_std, overheat_c):
    """Scatter predicted vs true basin temperature (mapped back to °C), with the
    overheat threshold drawn in. `pred_std`/`true_std` are standardized."""
    import matplotlib.pyplot as plt
    import numpy as np
    pred_c = pred_std * y_std + y_mean
    true_c = true_std * y_std + y_mean
    rmse_c = float(np.sqrt(np.mean((pred_c - true_c) ** 2)))

    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(true_c, pred_c, s=10, alpha=0.4, color="#6f7bf0", edgecolor="none")
    lo, hi = min(true_c.min(), pred_c.min()), max(true_c.max(), pred_c.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1, label="perfect")
    ax.axvline(overheat_c, color="#c44e52", lw=1.2, ls=":")
    ax.axhline(overheat_c, color="#c44e52", lw=1.2, ls=":", label=f"overheat {overheat_c:.0f}°C")
    ax.set_xlabel("true basin temperature (°C)")
    ax.set_ylabel("predicted (°C)")
    ax.set_title(f"Backup predictor — RMSE ≈ {rmse_c:.1f} °C")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"RMSE on the validation set: {rmse_c:.2f} °C")
