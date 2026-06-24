"""Presentation & interactivity helpers for the Block 2 notebook.

Everything in here is *plumbing* participants don't need to read: matplotlib /
pandas display settings, the heatmap drawing helper, and all the self-contained
HTML/CSS/JS widgets (data dictionary, quizzes, the column-role game, the
class-imbalance slider, the cost explorer, report cards, banners).

The notebook just does::

    import pdm_viz
    pdm_viz.configure_display()
    pdm_viz.data_dictionary()
    ...

Keeping it out of the notebook means the teaching cells stay about pandas, not
about CSS.
"""
import json as _json

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# The five physical sensor measurements — reused throughout the notebook.
NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Plain-language description of every column, shown by data_dictionary().
COLUMN_DOCS = {
    "UDI": "Unique Device Identifier — just a row counter (1, 2, 3, …). Pure bookkeeping, no real-world meaning.",
    "Product ID": "Serial number of the inspected part, e.g. 'L47181'. The leading letter repeats the machine grade.",
    "Type": "Machine quality grade: L (low), M (medium) or H (high). A category.",
    "Air temperature [K]": "Ambient air temperature around the machine, in Kelvin (≈ 300 K ≈ 27 °C).",
    "Process temperature [K]": "Temperature of the process itself, in Kelvin — runs a few degrees above the air.",
    "Rotational speed [rpm]": "Spindle rotation speed, in revolutions per minute.",
    "Torque [Nm]": "Twisting force the tool applies, in Newton-metres.",
    "Tool wear [min]": "Cumulative minutes the current tool has been used — it wears down over time.",
    "Machine failure": "1 if the machine failed during this cycle, 0 otherwise. ⭐ This is what we want to predict.",
    "Failure Type": "If it failed, HOW it failed (Heat Dissipation, Power, …). 'No Failure' when it didn't.",
    "Inspectors": "A packed inspection record (lead, team, shift, date, all_passed) stored as JSON text — nested data, not a real measurement.",
}


# --------------------------------------------------------------------- settings
def configure_display():
    """Apply the pandas + matplotlib display settings used throughout."""
    import pandas as pd
    pd.set_option("display.max_columns", 20)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    print("Display configured ✅")


# --------------------------------------------------------------------- heatmap
def heatmap(matrix, row_labels, col_labels, title, cmap="viridis", fmt=".2f"):
    """Annotated heatmap (used for the correlation matrix)."""
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(1.0 * len(col_labels) + 3, 0.6 * len(row_labels) + 3))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.grid(False)
    mid = np.nanmin(matrix) + 0.6 * (np.nanmax(matrix) - np.nanmin(matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, "—" if not np.isfinite(v) else format(v, fmt),
                    ha="center", va="center", fontsize=9,
                    color="white" if (np.isfinite(v) and v < mid) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontweight="bold"); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------- pandas anatomy
# A tiny toy table reused by the three "how a DataFrame works" demos below.
_TOY_COLS = ["Type", "Air temp", "Torque", "failure"]
_TOY_ROWS = [
    ["L", 298.1, 42.8, 0],
    ["M", 998.6, 46.3, 0],
    ["H", 300.4, -9.1, 1],
    ["L", 299.0, None, 0],   # a missing Torque reading — so .isna() has something to find
]


def _toy_table_html(col_class=None, row_class=None, cell_class=None):
    """Render the toy table; `*_class` are callables returning a CSS class (or '')."""
    head = "".join(
        '<th class="%s">%s</th>' % ((col_class(j) if col_class else ""), c)
        for j, c in enumerate(_TOY_COLS))
    body = ""
    for i, row in enumerate(_TOY_ROWS):
        cells = ""
        for j, v in enumerate(row):
            cls = ""
            if col_class:
                cls = col_class(j)
            if row_class:
                cls = row_class(i) or cls
            if cell_class:
                cls = cell_class(i, j) or cls
            txt = "NaN" if v is None else v
            cells += '<td class="%s">%s</td>' % (cls, txt)
        body += "<tr>%s</tr>" % cells
    return ("<table class='ana-tbl'><tr>" + head + "</tr>" + body + "</table>")


_ANA_CSS = '''
<style>
.ana-wrap{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;
    padding:16px;max-width:620px;background:#fff;color:#2b2d4f;margin:6px 0}
.ana-wrap h4{margin:0 0 4px;font-size:15px;font-weight:800;color:#2b2d6b}
.ana-wrap p{margin:0 0 12px;font-size:12.5px;color:#666}
.ana-tbl{border-collapse:collapse;font-size:12.5px;margin:6px 0}
.ana-tbl th,.ana-tbl td{border:1px solid #e2e5ef;padding:5px 11px;text-align:center}
.ana-tbl th{background:#f6f7fb;font-weight:700}
.ana-hi{background:#e7f0ff !important;box-shadow:inset 0 0 0 2px #6f7bf0}
.ana-hi2{background:#fff0e6 !important;box-shadow:inset 0 0 0 2px #dd8452}
.ana-code{font-family:ui-monospace,Menlo,monospace;background:#f3f0ff;border-radius:6px;
    padding:2px 7px;font-size:12.5px;color:#3b2d6b}
.ana-note{font-size:12px;color:#555;margin-top:8px;line-height:1.6}
.ana-note a{color:#6f53c0}
</style>'''


def column_select_demo():
    """Show that selecting columns keeps a *subset of columns, all rows*."""
    keep = {1, 2}  # Air temp, Torque
    tbl = _toy_table_html(col_class=lambda j: "ana-hi" if j in keep else "")
    display(HTML(_ANA_CSS + '''
<div class="ana-wrap">
  <h4>📐 Selecting columns</h4>
  <p><span class="ana-code">df[["Air temp", "Torque"]]</span> &nbsp;keeps those
     <b>columns</b> — and <b>every</b> row.</p>
  ''' + tbl + '''
  <div class="ana-note">One column? <span class="ana-code">df["Torque"]</span> →
  a <b>Series</b>. A list of names → a smaller DataFrame. The rows are untouched.</div>
</div>'''))


def row_filter_demo():
    """Symmetric to column_select_demo: boolean filtering keeps a *subset of rows,
    all columns*."""
    keep = {2}  # the failure==1 row
    tbl = _toy_table_html(row_class=lambda i: "ana-hi2" if i in keep else "")
    display(HTML(_ANA_CSS + '''
<div class="ana-wrap">
  <h4>📏 Filtering rows (boolean mask)</h4>
  <p><span class="ana-code">df[df["failure"] == 1]</span> &nbsp;keeps the
     <b>rows</b> where the condition is True — and <b>every</b> column.</p>
  ''' + tbl + '''
  <div class="ana-note">The condition <span class="ana-code">df["failure"] == 1</span>
  is itself a column of True/False the same length as the table — pandas keeps the
  True rows. This is the mirror image of selecting columns.</div>
</div>'''))


def axis_demo():
    """Show the two directions a pandas method can run: aggregate DOWN a column
    (many values → one number), or transform each cell (same shape back). Both
    sides show the *actual result* so the difference is concrete."""
    tbl_v = _toy_table_html(col_class=lambda j: "ana-hi" if j == 1 else "")
    tbl_h = _toy_table_html(col_class=lambda j: "ana-hi2" if j == 2 else "")

    # aggregate result: min of the Air temp column
    air_vals = [r[1] for r in _TOY_ROWS]
    air_min = min(air_vals)

    # transform result: Torque -> Torque.isna(), one True/False per row
    torque_vals = [r[2] for r in _TOY_ROWS]
    isna_rows = "".join(
        '<tr><td>%s</td><td><b style="color:%s">%s</b></td></tr>'
        % (("NaN" if v is None else v), ("#b23b3b" if v is None else "#1f7a43"), (v is None))
        for v in torque_vals)
    isna_tbl = ('<table class="ana-tbl"><tr><th>Torque</th><th>.isna()</th></tr>'
                + isna_rows + '</table>')

    display(HTML(_ANA_CSS + '''
<div class="ana-wrap">
  <h4>🧭 Which way does the function run?</h4>
  <p>Most pandas methods either <b>aggregate down a column</b> (many values → one)
     or <b>transform each cell</b> (same shape back). Watch the result on the right of each.</p>
  <div style="display:flex;gap:18px;flex-wrap:wrap">
    <div>
      <div style="font-size:12.5px;font-weight:700;color:#3b46b0">⬇ aggregate (per column)</div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        ''' + tbl_v + '''
        <div style="font-size:18px;color:#b9a9e6">➜</div>
        <div class="ana-code" style="font-size:16px"><b>''' + ("%g" % air_min) + '''</b></div>
      </div>
      <div class="ana-note"><span class="ana-code">df["Air temp"].min()</span>
      collapses the whole column to <b>one number</b> (here ''' + ("%g" % air_min) + ''').
      Same for <span class="ana-code">.max()</span>, <span class="ana-code">.mean()</span>.</div>
    </div>
    <div>
      <div style="font-size:12.5px;font-weight:700;color:#b5642f">➡ transform (per cell)</div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        ''' + tbl_h + '''
        <div style="font-size:18px;color:#e0b48f">➜</div>
        ''' + isna_tbl + '''
      </div>
      <div class="ana-note"><span class="ana-code">df["Torque"].isna()</span> →
      a True/False for <b>every</b> cell (same shape back); nothing is collapsed.</div>
    </div>
  </div>
  <div class="ana-note">📚 pandas docs:
    <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.min.html">min</a> ·
    <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.max.html">max</a> ·
    <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html">mean</a> ·
    <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html">isna</a>
  </div>
</div>'''))


def predict_unique_count():
    """Interactive predict-the-output for .unique() / .nunique() / .count() /
    .value_counts(), on two tiny columns — one clean, one with a missing value, so
    the difference in how each method treats NaN becomes concrete.

    The four **numeric** calls get input boxes the student must fill in (checked in
    the browser — green/red, no answer shown); the three **list / per-value** calls
    stay hidden behind a Reveal so the student commits to a guess first."""
    def col_table(name, vals):
        body = "".join(
            "<tr><td>%s</td></tr>"
            % ("<i style='color:#b23b3b'>NaN</i>" if v is None else v) for v in vals)
        return "<table class='ana-tbl'><tr><th>%s</th></tr>%s</table>" % (name, body)

    t1 = col_table("grade", ["A", "B", "A", "A", "C"])
    t2 = col_table("status", ["ok", "ok", None, "bad", "ok"])

    # numeric calls -> the student types the number; checked client-side
    numeric = [
        ("grade.nunique()", 3),
        ("grade.count()", 5),
        ("status.count()", 4),
        ("status.nunique()", 2),
    ]
    # list / per-value calls -> revealed only after committing to a guess
    reveal = [
        ("grade.unique()", "['A', 'B', 'C'] — the distinct values"),
        ("grade.value_counts()", "A → 3, B → 1, C → 1"),
        ("status.unique()", "['ok', nan, 'bad'] — NaN <b>is</b> listed"),
        ("status.value_counts()", "ok → 3, bad → 1 — NaN dropped by default"),
    ]
    data = [{"a": a} for _, a in numeric]
    uid = "puc_" + str(abs(hash(tuple(q for q, _ in numeric + reveal))) % 10**8)

    num_rows = "".join(
        '<tr><td class="puc-call"><span class="ana-code">%s</span></td>'
        '<td><input class="puc-in" type="number" data-i="%d" placeholder="?"></td>'
        '<td class="puc-mark" data-i="%d"></td></tr>'
        % (q, i, i) for i, (q, _) in enumerate(numeric))
    rev_rows = "".join(
        '<tr><td class="puc-call"><span class="ana-code">%s</span></td>'
        '<td class="puc-ans">%s</td></tr>' % (q, a) for q, a in reveal)
    html = _ANA_CSS + '''
<style>
#__UID__ .puc-tbl{border-collapse:collapse;font-size:13px;width:100%;margin-top:4px}
#__UID__ .puc-tbl td{border:1px solid #eef0f6;padding:6px 10px}
#__UID__ .puc-call{white-space:nowrap}
#__UID__ .puc-cap{font-size:12.5px;font-weight:700;color:#3b46b0;margin:12px 0 2px}
#__UID__ .puc-cap2{font-size:12.5px;font-weight:700;color:#b5642f;margin:14px 0 2px}
#__UID__ .puc-in{width:90px;padding:6px 8px;border:1px solid #c2c7da;border-radius:7px;font-size:13px}
#__UID__ .puc-in.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .puc-in.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .puc-mark{width:1%;white-space:nowrap;font-size:14px;font-weight:700}
#__UID__ .puc-mark.ok{color:#1f7a43}
#__UID__ .puc-mark.no{color:#b23b3b}
#__UID__ .puc-ans{color:#1f7a43;display:none}
#__UID__.shown .puc-ans{display:table-cell}
#__UID__ .puc-btn{cursor:pointer;border:none;border-radius:8px;padding:8px 16px;font-size:13px;
    font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin:12px 8px 0 0}
#__UID__ .puc-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:8px;min-height:18px}
</style>
<div class="ana-wrap" id="__UID__">
  <h4>🔮 Predict the output</h4>
  <p>Two tiny columns. Work each call out <b>in your head first</b> — watch how
     <span class="ana-code">count</span>, <span class="ana-code">value_counts</span> and
     <span class="ana-code">nunique</span> quietly skip the missing value, while
     <span class="ana-code">unique</span> keeps it.</p>
  <div style="display:flex;gap:22px;flex-wrap:wrap;margin-bottom:6px">''' + t1 + t2 + '''</div>
  <div class="puc-cap">① These return a single number — type your prediction, then check:</div>
  <table class="puc-tbl">''' + num_rows + '''</table>
  <button class="puc-btn puc-check">Check my numbers</button>
  <div class="puc-status"></div>
  <div class="puc-cap2">② These return a list / per-value count — guess out loud, then reveal:</div>
  <table class="puc-tbl">''' + rev_rows + '''</table>
  <button class="puc-btn puc-reveal">Reveal these answers</button>
</div>
<script>
(function(){
  const DATA=__DATA__, root=document.getElementById("__UID__");
  root.querySelector(".puc-check").addEventListener("click",()=>{
    let right=0, answered=0;
    root.querySelectorAll(".puc-in").forEach(inp=>{
      const i=+inp.dataset.i, want=DATA[i].a;
      const mark=root.querySelector('.puc-mark[data-i="'+i+'"]');
      inp.classList.remove("ok","no"); mark.classList.remove("ok","no");
      if(inp.value===""){ mark.textContent="•"; return; }   // no guess yet — stay neutral
      answered++;
      const ok=(parseInt(inp.value,10)===want); if(ok)right++;
      inp.classList.add(ok?"ok":"no"); mark.classList.add(ok?"ok":"no");
      mark.textContent = ok ? "✓ correct" : "✗ try again";
    });
    const status=root.querySelector(".puc-status");
    if(answered===0){ status.textContent="Type a prediction in each box first."; return; }
    status.textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉":" — look again at the NaN row!");
  });
  root.querySelector(".puc-reveal").addEventListener("click",()=>{
    root.classList.add("shown");
    root.querySelector(".puc-reveal").textContent="Answers revealed 👇";
  });
})();
</script>'''
    display(HTML(html.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))))


# --------------------------------------------------------------------- banners
def pipeline_diagram():
    """The 'our investigation' step diagram."""
    display(HTML('''
<style>
.pl{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:linear-gradient(135deg,#eef4ff,#f7eefc);
    border-radius:18px;padding:20px 16px;margin:6px 0;border:1px solid #e6e8ff;color:#2c2350}
.pl-h{font-size:19px;font-weight:800;color:#2b2d6b;margin:0 0 14px}
.pl-row{display:flex;align-items:center;flex-wrap:wrap;gap:0;justify-content:center}
.pl-step{text-align:center;padding:0 6px;min-width:96px}
.pl-ic{width:48px;height:48px;border-radius:50%;margin:0 auto 6px;display:flex;align-items:center;
       justify-content:center;font-size:22px;color:#fff;box-shadow:0 5px 12px rgba(102,126,234,.3)}
.pl-t{font-weight:700;font-size:12.5px;color:#2c2350}
.pl-ar{font-size:20px;color:#b9a9e6;padding:0 2px}
</style>
<div class="pl"><div class="pl-h">🧭 Our investigation</div><div class="pl-row">
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#667eea,#7a8cf0)">🔎</div><div class="pl-t">Inspect</div></div>
<div class="pl-ar">➜</div>
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#7a8cf0,#8f7ae6)">🧹</div><div class="pl-t">Find problems</div></div>
<div class="pl-ar">➜</div>
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#8f7ae6,#a86fd6)">📊</div><div class="pl-t">Visualize</div></div>
<div class="pl-ar">➜</div>
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#a86fd6,#c06fc2)">🛠️</div><div class="pl-t">Prepare</div></div>
<div class="pl-ar">➜</div>
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#c06fc2,#d76faa)">🤖</div><div class="pl-t">Train</div></div>
<div class="pl-ar">➜</div>
<div class="pl-step"><div class="pl-ic" style="background:linear-gradient(135deg,#d76faa,#e86f90)">⚖️</div><div class="pl-t">Judge</div></div>
</div></div>'''))


def engineers_pipeline():
    """The engineering team's actual pipeline, as a block diagram with the
    foreshadowing hints we'll return to. Replaces dumping their code up front."""
    steps = [
        ("📄", "Load the CSV", "as-is, every column", "Did they even look at it?"),
        ("🩹", "Fill the gaps", "missing → column median", "Trust those values?"),
        ("🏷️", "One-hot encode", "<b>every</b> column, incl. <code>Failure&nbsp;Type</code>", "Available at inference?"),
        ("🤖", "Train LogReg", "75% train / 25% test", "Same algorithm we'll use"),
        ("🎉", "Score it", "<b>≈ 100%</b> accuracy", "Too good to be true?"),
    ]
    blocks = ""
    for i, (ic, t, sub, hint) in enumerate(steps):
        blocks += (
            '<div class="ep-step"><div class="ep-ic">%s</div>'
            '<div class="ep-t">%s</div><div class="ep-sub">%s</div>'
            '<div class="ep-hint">%s</div></div>' % (ic, t, sub, hint))
        if i < len(steps) - 1:
            blocks += '<div class="ep-ar">➜</div>'
    display(HTML('''
<style>
.ep{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#fff;border:1px solid #f0c8c8;
    border-radius:16px;padding:18px 14px;margin:6px 0;color:#2c2350}
.ep-h{font-size:16px;font-weight:800;color:#b23b3b;margin:0 0 4px}
.ep-s{font-size:12.5px;color:#777;margin:0 0 14px}
.ep-row{display:flex;align-items:stretch;flex-wrap:wrap;gap:0;justify-content:center}
.ep-step{flex:1 1 120px;min-width:118px;max-width:170px;text-align:center;background:#fbfbff;
    border:1px solid #ececf6;border-radius:12px;padding:10px 8px}
.ep-ic{font-size:24px;margin-bottom:4px}
.ep-t{font-weight:700;font-size:13px;color:#2c2350}
.ep-sub{font-size:11.5px;color:#555;margin-top:2px;min-height:28px}
.ep-sub code{background:#f1edff;border-radius:4px;padding:0 3px}
.ep-hint{font-size:10.5px;color:#c06070;font-style:italic;margin-top:6px;border-top:1px dashed #f0d0d0;padding-top:5px}
.ep-ar{display:flex;align-items:center;font-size:18px;color:#d9a9b9;padding:0 2px}
</style>
<div class="ep"><div class="ep-h">⚙️ The engineering team's pipeline</div>
<div class="ep-s">This is everything they did to get to "ready to ship". We'll open up
each block in turn — the italic notes are the questions we'll answer.</div>
<div class="ep-row">''' + blocks + '''</div></div>'''))


def bait_banner():
    """Red 'near-perfect score is suspicious' callout under the engineers' result.

    Colab's dark theme injects a `color: … !important` rule on output text, which
    overrides a plain inline `color`. We beat it with a scoped `!important` rule on
    the card *and every child* so the dark text stays readable on the light card."""
    display(HTML('''
<style>
.bait-banner, .bait-banner * { color:#2b2b2b !important; }
.bait-banner { background:#fff4f4 !important; }
</style>
<div class="bait-banner" style="font-size:15px;padding:10px 14px;background:#fff4f4;
     border-left:5px solid #e74c3c;border-radius:6px;color:#2b2b2b">
<b>🚨 A near-perfect score.</b> In the real world, a model that predicts rare machine
failures almost perfectly is <i>far</i> more likely to be <b>broken</b> than brilliant.
Let\'s find out why.</div>'''))


def checklist():
    """Final 'before you trust it' checklist card."""
    display(HTML('''
<div style="font-family:system-ui;max-width:640px;border:1px solid #e6e8ee;border-radius:14px;overflow:hidden;background:#fff;color:#333">
<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:12px 16px;font-weight:800;font-size:15px">
✅ Before trusting any ML result, check…</div>
<ul style="margin:0;padding:14px 26px;font-size:13.8px;line-height:1.9;color:#333">
<li><b>What is one row?</b> Know your observation, features, and target.</li>
<li><b>Leakage</b> — any feature that encodes the answer or won't exist at prediction time? (<i>Failure Type</i>)</li>
<li><b>Artifacts</b> — IDs / keys masquerading as features? (<i>UDI, Product ID</i>)</li>
<li><b>Missing, duplicate, impossible</b> values — and handle them deliberately.</li>
<li><b>Inconsistent categories</b> — one concept, many spellings.</li>
<li><b>Class balance</b> — is "accuracy" even meaningful here?</li>
<li><b>Right metric</b> — precision / recall / F1, weighted by the <b>cost</b> of each error.</li>
<li><b>Distribution shift</b> — does production data look like training data?</li>
</ul></div>'''))


# --------------------------------------------------------------------- data dictionary
def data_dictionary(columns=None):
    """Interactive data dictionary: click a column to reveal what it means."""
    columns = list(columns) if columns is not None else list(COLUMN_DOCS)
    docs = {c: COLUMN_DOCS.get(c, "(no description available)") for c in columns}
    uid = "dd_" + str(abs(hash(tuple(columns))) % 10**8)
    chips = "".join('<span class="dd-chip" data-col="%s">%s</span>' % (c, c) for c in columns)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:760px;background:#fff;color:#2c2350}
#__UID__ .dd-head{font-weight:800;font-size:15px;margin-bottom:10px}
#__UID__ .dd-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .dd-cols{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px}
#__UID__ .dd-chip{cursor:pointer;border:1px solid #d7dae6;background:#f6f7fb;border-radius:999px;padding:6px 11px;font-size:12.5px;transition:.15s}
#__UID__ .dd-chip:hover{border-color:#764ba2}
#__UID__ .dd-chip.sel{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-color:transparent}
#__UID__ .dd-panel{min-height:54px;background:#f3f0ff;border-radius:10px;padding:12px 14px;font-size:13.5px;color:#2c2350}
#__UID__ .dd-panel b{color:#3b2d6b}
</style>
<div id="__UID__">
  <div class="dd-head">📖 Data dictionary — click a column you don't recognise</div>
  <div class="dd-sub">Poll the columns one by one to learn what each one means.</div>
  <div class="dd-cols">__CHIPS__</div>
  <div class="dd-panel">👆 Click any column above to see its description.</div>
</div>
<script>
(function(){
  const DOCS=__DOCS__, root=document.getElementById("__UID__");
  const chips=root.querySelectorAll(".dd-chip"), panel=root.querySelector(".dd-panel");
  chips.forEach(c=>c.addEventListener("click",()=>{
    chips.forEach(x=>x.classList.remove("sel")); c.classList.add("sel");
    panel.innerHTML="<b>"+c.dataset.col+"</b> — "+DOCS[c.dataset.col];
  }));
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__CHIPS__", chips)
            .replace("__DOCS__", _json.dumps(docs)))
    display(HTML(html))


# --------------------------------------------------------------------- quizzes
def _tf_render(title, statements, prompt="Click every statement you think is TRUE, then check."):
    """A shuffled true/false quiz.

    `statements` is a list of (text, is_true). Pass an equal number of true and
    false statements. Order is shuffled in the browser on every load.
    """
    items = [{"t": t, "ok": bool(ok)} for t, ok in statements]
    uid = "tf_" + str(abs(hash(tuple((s["t"], s["ok"]) for s in items))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:760px;background:#fff;color:#2c2350}
#__UID__ .tf-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .tf-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .tf-row{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;transition:.12s}
#__UID__ .tf-row:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .tf-row .box{width:18px;height:18px;border-radius:5px;border:2px solid #c2c7da;flex:0 0 auto}
#__UID__ .tf-row.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .tf-row.sel .box{background:#764ba2;border-color:#764ba2}
#__UID__ .tf-row.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .tf-row.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .tf-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .tf-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="tf-head">__TITLE__</div>
  <div class="tf-sub">__PROMPT__</div>
  <div class="tf-list"></div>
  <button class="tf-btn">Check my answers</button>
  <div class="tf-status"></div>
</div>
<script>
(function(){
  let DATA=__DATA__.slice();
  for(let i=DATA.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[DATA[i],DATA[j]]=[DATA[j],DATA[i]];}
  const root=document.getElementById("__UID__"), list=root.querySelector(".tf-list");
  const status=root.querySelector(".tf-status");
  DATA.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="tf-row"; row.dataset.i=i;
    row.innerHTML='<div class="box"></div><div>'+d.t+'</div>';
    row.addEventListener("click",()=>{row.classList.remove("ok","no");row.classList.toggle("sel");});
    list.appendChild(row);
  });
  root.querySelector(".tf-btn").addEventListener("click",()=>{
    let right=0; const rows=list.querySelectorAll(".tf-row");
    rows.forEach(r=>{
      const d=DATA[+r.dataset.i], sel=r.classList.contains("sel");
      r.classList.remove("ok","no");
      const correct=(sel===d.ok); if(correct)right++;
      r.classList.add(correct?"ok":"no");
    });
    status.textContent = right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉 — nailed it!":" — green = right call, red = rethink it.");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__DATA__", _json.dumps(items)))
    display(HTML(html))


def _nq_render(title, questions, prompt="Run your own pandas queries to find each number, then check."):
    """Numeric-answer quiz. `questions` is a list of (prompt, integer_answer).

    Students must slice the DataFrame themselves to find each value — the widget
    only tells them whether the number is right.
    """
    data = [{"q": q, "a": int(a)} for q, a in questions]
    uid = "nq_" + str(abs(hash(tuple((d["q"], d["a"]) for d in data))) % 10**8)
    rows = "".join(
        '<div class="nq-row"><div class="nq-q">%s</div>'
        '<input class="nq-in" type="number" data-i="%d" placeholder="?"></div>' % (d["q"], i)
        for i, d in enumerate(data))
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:760px;background:#fff;color:#2c2350}
#__UID__ .nq-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .nq-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .nq-row{display:flex;align-items:center;gap:12px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px}
#__UID__ .nq-q{flex:1;font-size:13.5px}
#__UID__ .nq-in{width:110px;padding:7px 9px;border:1px solid #c2c7da;border-radius:8px;font-size:13.5px}
#__UID__ .nq-in.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .nq-in.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .nq-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .nq-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="nq-head">__TITLE__</div>
  <div class="nq-sub">__PROMPT__</div>
  __ROWS__
  <button class="nq-btn">Check my answers</button>
  <div class="nq-status"></div>
</div>
<script>
(function(){
  const DATA=__DATA__, root=document.getElementById("__UID__");
  root.querySelector(".nq-btn").addEventListener("click",()=>{
    let right=0; root.querySelectorAll(".nq-in").forEach(inp=>{
      const want=DATA[+inp.dataset.i].a, got=parseInt(inp.value,10);
      inp.classList.remove("ok","no");
      const ok=(got===want); if(ok)right++; inp.classList.add(ok?"ok":"no");
    });
    root.querySelector(".nq-status").textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉":" — keep slicing!");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__ROWS__", rows)
            .replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


# --------------------------------------------------------------------- column-role game
def column_role_game(answer=None):
    """Tag-each-column-with-its-role mini game. The answer key is baked in (below)
    so it never appears in the notebook cell the student can read."""
    answer = answer if answer is not None else _ROLE_ANSWERS
    uid = "rg_" + str(abs(hash(tuple(answer))) % 10**8)
    chips = "".join('<span class="rg-chip" data-col="%s">%s</span>' % (c, c) for c in answer)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:760px;background:#fff;color:#2c2350}
#__UID__ .rg-head{font-weight:800;font-size:15px;margin-bottom:12px}
#__UID__ .rg-cols{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px}
#__UID__ .rg-chip{cursor:pointer;border:1px solid #d7dae6;background:#f6f7fb;border-radius:999px;padding:6px 11px;font-size:12.5px;transition:.15s}
#__UID__ .rg-chip:hover{border-color:#764ba2}
#__UID__ .rg-chip.sel{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border-color:transparent}
#__UID__ .rg-chip.ok{background:#e7f7ec;border-color:#46b46e;color:#1f7a43}
#__UID__ .rg-chip.no{background:#fdecec;border-color:#e07a7a;color:#b23b3b}
#__UID__ .rg-roles{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
#__UID__ .rg-roles button{cursor:pointer;border:1px solid #cfd3e2;background:#fff;border-radius:8px;padding:7px 12px;font-size:13px;font-weight:600}
#__UID__ .rg-roles button:hover{background:#f1edff;border-color:#764ba2}
#__UID__ .rg-status{font-size:13px;color:#555;min-height:18px}
#__UID__ .rg-score{font-size:13px;font-weight:800;color:#3b2d6b;margin-top:6px}
</style>
<div id="__UID__">
  <div class="rg-head">🏷️ Tag each column with its role</div>
  <div class="rg-cols">__CHIPS__</div>
  <div class="rg-roles">
    <button data-r="feature">feature</button>
    <button data-r="target">target</button>
    <button data-r="id">ID / artifact</button>
    <button data-r="leakage">leakage</button>
  </div>
  <div class="rg-status">Pick a column, then click its role.</div>
  <div class="rg-score"></div>
</div>
<script>
(function(){
  const ANS=__ANS__, root=document.getElementById("__UID__");
  const chips=root.querySelectorAll(".rg-chip"), roles=root.querySelectorAll(".rg-roles button");
  const status=root.querySelector(".rg-status"), score=root.querySelector(".rg-score");
  let sel=null; const done={};
  chips.forEach(c=>c.addEventListener("click",()=>{
    sel=c.dataset.col; chips.forEach(x=>x.classList.remove("sel")); c.classList.add("sel");
    status.textContent='Column "'+sel+'" selected — what is it?';
  }));
  roles.forEach(b=>b.addEventListener("click",()=>{
    if(!sel){status.textContent="Pick a column first!";return;}
    const ok=ANS[sel]===b.dataset.r;
    const chip=root.querySelector('.rg-chip[data-col="'+CSS.escape(sel)+'"]');
    chip.classList.remove("ok","no"); chip.classList.add(ok?"ok":"no");
    chip.textContent=sel+(ok?" ✅":" ❌"); done[sel]=ok;
    const good=Object.values(done).filter(Boolean).length;
    status.textContent= ok ? "Correct!" : "Not quite — rethink “"+sel+"”.";
    score.textContent= good+" / "+Object.keys(ANS).length+" correct";
    sel=null; chips.forEach(x=>x.classList.remove("sel"));
  }));
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__CHIPS__", chips)
            .replace("__ANS__", _json.dumps(answer)))
    display(HTML(html))


# --------------------------------------------------------------------- multiple choice
def _mc_render(title, question, options, answer_index, reveal):
    """Single-select multiple-choice widget. Reveals the correct option and an
    explanation only AFTER the student commits to an answer and clicks check."""
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = "mc_" + str(abs(hash((question, tuple(options), answer_index))) % 10**8)
    rows = "".join(
        '<div class="mc-opt" data-i="%d"><span class="mc-dot"></span>%s</div>' % (i, o)
        for i, o in enumerate(options))
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:760px;background:#fff;color:#2c2350}
#__UID__ .mc-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .mc-q{color:#444;font-size:13.5px;margin-bottom:12px}
#__UID__ .mc-opt{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;transition:.12s}
#__UID__ .mc-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .mc-dot{width:16px;height:16px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto}
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


# --------------------------------------------------------------------- match operations
def _ops_render(title, items, prompt="Click the operation that fits each situation, then check all."):
    """Scenario → operation matcher. `items` is a list of (scenario, options, answer_index).
    Each scenario is a single-select; one button grades them all at once."""
    data = [{"q": q, "opts": list(opts), "ans": int(ans)} for q, opts, ans in items]
    uid = "op_" + str(abs(hash(tuple((d["q"], tuple(d["opts"]), d["ans"]) for d in data))) % 10**8)
    blocks = ""
    for bi, d in enumerate(data):
        opts = "".join(
            '<div class="op-opt" data-b="%d" data-i="%d"><code>%s</code></div>' % (bi, oi, o)
            for oi, o in enumerate(d["opts"]))
        blocks += ('<div class="op-block" data-b="%d"><div class="op-q">%d. %s</div>'
                   '<div class="op-opts">%s</div></div>' % (bi, bi + 1, d["q"], opts))
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff;color:#2c2350}
#__UID__ .op-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .op-sub{color:#666;font-size:12.5px;margin-bottom:14px}
#__UID__ .op-block{border-top:1px solid #eef0f6;padding:12px 0}
#__UID__ .op-q{font-size:13.5px;font-weight:600;margin-bottom:8px;color:#2c2350}
#__UID__ .op-opts{display:flex;flex-direction:column;gap:6px}
#__UID__ .op-opt{border:1px solid #e2e5ef;border-radius:9px;padding:7px 11px;cursor:pointer;font-size:13px;transition:.12s}
#__UID__ .op-opt code{background:#f3f0ff;border-radius:5px;padding:1px 5px}
#__UID__ .op-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .op-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .op-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .op-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .op-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:12px}
#__UID__ .op-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="op-head">__TITLE__</div>
  <div class="op-sub">__PROMPT__</div>
  __BLOCKS__
  <button class="op-btn">Check all</button>
  <div class="op-status"></div>
</div>
<script>
(function(){
  const DATA=__DATA__, root=document.getElementById("__UID__");
  const sel={};
  root.querySelectorAll(".op-opt").forEach(o=>o.addEventListener("click",()=>{
    const b=+o.dataset.b;
    root.querySelectorAll('.op-opt[data-b="'+b+'"]').forEach(x=>x.classList.remove("sel","ok","no"));
    o.classList.add("sel"); sel[b]=+o.dataset.i;
  }));
  root.querySelector(".op-btn").addEventListener("click",()=>{
    let right=0;
    DATA.forEach((d,b)=>{
      root.querySelectorAll('.op-opt[data-b="'+b+'"]').forEach(o=>{
        o.classList.remove("sel","ok","no");
        const i=+o.dataset.i;
        if(i===d.ans)o.classList.add("ok");
        else if(i===sel[b])o.classList.add("no");
      });
      if(sel[b]===d.ans)right++;
    });
    root.querySelector(".op-status").textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉 — you've got the toolkit!":" — green = the right tool.");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__BLOCKS__", blocks)
            .replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


# --------------------------------------------------------------------- quiz content
# Answer keys live HERE (in the helper module students don't read), never in the
# notebook cell — so the quizzes can't be solved by peeking at the source.
_ROLE_ANSWERS = {
    "UDI": "id",
    "Product ID": "id",
    "Type": "feature",
    "Air temperature [K]": "feature",
    "Process temperature [K]": "feature",
    "Rotational speed [rpm]": "feature",
    "Torque [Nm]": "feature",
    "Tool wear [min]": "feature",
    "Machine failure": "target",
    "Failure Type": "leakage",
}

_TF_QUIZZES = {
    "describe": ("What can we read off the .describe() table?", [
        ("Two sensor columns have fewer than 8150 values, so they contain missing data.", True),
        ("The maximum `Air temperature [K]` (≈ 999) is physically impossible — a faulty sensor.", True),
        ("`No Failure` is by far the most frequent value of `Failure Type`.", True),
        ("Every column has the full 8150 non-null values.", False),
        ("`Machine failure` has a mean near 0.5, so the two classes are balanced.", False),
        ("`Type` has exactly the three expected categories L, M and H.", False)]),
    "histograms": ("Reading the sensor histograms", [
        ("Air temperature looks like one spike only because a few ~999 K outliers stretch the "
         "x-axis — the real readings are squeezed together.", True),
        ("Rotational speed is right-skewed, with a long tail toward higher speeds.", True),
        ("The impossible values from Task 8 show up as tiny bars far from the main cluster.", True),
        ("Air temperature is truly constant: every machine runs at one identical temperature.", False),
        ("None of the sensors show any outliers.", False),
        ("Torque readings are all negative.", False)]),
    "scaling": ("Why did the engineers scale the features — and why must we too?", [
        ("The features span wildly different ranges (rpm in the thousands, torque in the tens), "
         "so without scaling the large-range feature dominates the model's coefficients.", True),
        ("StandardScaler centers each feature to mean 0 and unit variance, putting them on "
         "comparable footing.", True),
        ("The scaler must be fit on the TRAINING split only; fitting it on all the data leaks "
         "test information into training.", True),
        ("Scaling changes which rows are labelled failures versus healthy.", False),
        ("A decision tree's splits would be broken unless you scale the features first.", False),
        ("Scaling is purely cosmetic and never affects a logistic-regression model's predictions.", False)]),
    "correlation": ("Do you read correlation right? (a couple are guesses about THIS data)", [
        ("A correlation near 0 means the two features have no *linear* relationship.", True),
        ("Correlations of +0.9 and −0.9 are equally strong — the sign only gives the direction.", True),
        ("Rotational speed and torque are likely strongly related — they trade off in mechanical power.", True),
        ("A correlation of −0.85 is weak, because it is negative.", False),
        ("A strong correlation between two features proves that one *causes* the other.", False),
        ("Two columns must be strongly correlated whenever they measure the same kind of "
         "quantity (e.g. both temperatures).", False)]),
}

_NUM_QUIZZES = {
    "exploration": ("Answer these by slicing the DataFrame yourself", [
        ("How many rows have a Tool wear above 200 minutes?", 563),
        ("How many rows have a missing (NaN) Torque value?", 161),
        ("How many rows have Failure Type 'Heat Dissipation Failure'?", 95)]),
    "always_healthy": ("Predict first: how accurate is an 'always healthy' model?", [
        ("If 50% of rows are failures (50% healthy), what % accuracy does it score?", 50),
        ("If 25% of rows are failures (75% healthy)?", 75),
        ("If 0% of rows are failures (100% healthy)?", 100)]),
}

_MC_QUIZZES = {
    "inspectors_useful": (
        "🧐 Is this column useful for the model?",
        "You've unpacked <code>Inspectors</code> into its keys "
        "(<code>lead</code>, <code>team</code>, <code>shift</code>, <code>date</code>, "
        "<code>all_passed</code>). The model's job is to predict failure from a machine's "
        "<b>sensor readings</b>. Should these inspection fields be features?",
        ["Yes — the inspector who signed off plausibly influences whether a machine fails.",
         "Yes — the <code>date</code> pins down exactly when each failure occurred.",
         "No — these are administrative bookkeeping fields, not measurements of the machine.",
         "Unclear — we'd have to train a model on them first to find out."],
        2,
        "These are administrative records — <i>who</i> inspected the machine and <i>when</i>, "
        "not anything about how the machine is actually running. None of it is a measurement "
        "that helps predict failure, and identifiers like inspector names invite the model to "
        "memorise noise. We'll <b>drop</b> it from the modeling table."),
}


# Scenario → operation matcher shown at the end of Part 1. (scenario, options, answer_index)
_OPS_QUIZ = [
    ("List the distinct machine grades in the `Type` column.",
     ['historical_df["Type"].unique()',
      'historical_df["Type"].value_counts()',
      'historical_df["Type"].count()',
      'historical_df["Type"].isna()'], 0),
    ("Count how many rows are missing a Torque reading.",
     ['historical_df["Torque [Nm]"].count()',
      'historical_df["Torque [Nm]"].isna()',
      'historical_df["Torque [Nm]"].isna().sum()',
      'historical_df["Torque [Nm]"].unique()'], 2),
    ("Keep only the rows where the machine actually failed.",
     ['historical_df["Machine failure"] == 1',
      'historical_df[historical_df["Machine failure"] == 1]',
      'historical_df[["Machine failure"]]',
      'historical_df["Machine failure"].value_counts()'], 1),
    ("Find how often each Failure Type occurs.",
     ['historical_df["Failure Type"].unique()',
      'historical_df["Failure Type"].nunique()',
      'historical_df["Failure Type"].value_counts()',
      'historical_df["Failure Type"].count()'], 2),
    ("Get the average Torque across all rows, as one number.",
     ['historical_df["Torque [Nm]"].describe()',
      'historical_df["Torque [Nm]"].mean()',
      'historical_df["Torque [Nm]"].value_counts()',
      'historical_df["Torque [Nm]"].unique()'], 1),
]


def operations_quiz():
    """Render the end-of-Part-1 'pick the right pandas move' matcher."""
    _ops_render("🧰 Which pandas operation fits?", _OPS_QUIZ)


def true_false_quiz(key):
    """Render a baked-in true/false quiz by name (answers hidden in this module)."""
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)


def number_quiz(key):
    """Render a baked-in numeric quiz by name (answers hidden in this module)."""
    title, questions = _NUM_QUIZZES[key]
    _nq_render(title, questions)


def mc_quiz(key):
    """Render a baked-in single-select multiple-choice quiz by name."""
    _mc_render(*_MC_QUIZZES[key])


# --------------------------------------------------------------------- report card
def quality_report_card(missing, duplicates, type_spellings, impossible,
                        leakage=1, id_like=2):
    """Render the data-quality report card from student-computed numbers."""
    items = [
        ("Missing values", int(missing), "cells need imputing"),
        ("Duplicate rows", int(duplicates), "drop before splitting"),
        ("Inconsistent `Type` spellings", int(type_spellings), "extra spellings of L/M/H"),
        ("Impossible sensor values", int(impossible), "negative / zero / stuck"),
        ("Leakage columns", int(leakage), "`Failure Type` — remove"),
        ("ID-like columns", int(id_like), "`UDI`, `Product ID` — remove"),
    ]
    rows = "".join(
        '<tr><td style="padding:7px 12px">%s</td>'
        '<td style="padding:7px 12px;text-align:center;font-weight:800;color:#b23b3b">%s</td>'
        '<td style="padding:7px 12px;color:#777;font-size:12.5px">%s</td></tr>' % it
        for it in items)
    display(HTML(
        '<div style="font-family:system-ui;max-width:560px;border:1px solid #eee;border-radius:12px;overflow:hidden;background:#fff;color:#2c2350">'
        '<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:11px 14px;font-weight:800">'
        '🩺 Data Quality Report — historical dataset</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:13.5px">' + rows + '</table></div>'))


# --------------------------------------------------------------------- imbalance slider
def imbalance_slider(actual_failure_pct):
    """Slider showing how an 'always healthy' classifier's accuracy inflates as
    failures get rarer. The dataset's real failure share is marked."""
    uid = "im_" + str(abs(hash(("im", round(float(actual_failure_pct), 4)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:640px;background:#fff;color:#2c2350}
#__UID__ .im-head{font-weight:800;font-size:15px;margin-bottom:10px}
#__UID__ label{display:block;font-size:13px;margin:10px 0;color:#333}
#__UID__ input[type=range]{width:100%}
#__UID__ .im-acc{margin-top:8px;font-size:15px;background:#f3f0ff;border-radius:8px;padding:12px 14px;line-height:1.6}
#__UID__ .im-big{font-size:24px;font-weight:800;color:#3b2d6b}
#__UID__ .im-note{font-size:12.5px;color:#777;margin-top:6px}
</style>
<div id="__UID__">
  <div class="im-head">🩻 The "always healthy" classifier — how accuracy inflates</div>
  <label>Share of rows that are real failures: <b><span class="im-pv">3.4</span>%</b>
    <input type="range" class="im-p" min="0.5" max="50" step="0.5" value="__ACT__"></label>
  <div class="im-acc">
    A model that <b>always predicts "healthy"</b> would score
    <span class="im-big"><span class="im-av"></span>%</span> accuracy<br>
    …while catching <b>0</b> of the failures (recall = 0%).
  </div>
  <div class="im-note"></div>
</div>
<script>
(function(){
  const ACT=__ACT__, root=document.getElementById("__UID__");
  const p=root.querySelector(".im-p");
  function upd(){
    const share=+p.value, acc=100-share;
    root.querySelector(".im-pv").textContent=share.toFixed(1);
    root.querySelector(".im-av").textContent=acc.toFixed(1);
    const note=root.querySelector(".im-note");
    if(Math.abs(share-ACT)<0.26){
      note.innerHTML="⬅️ This is roughly <b>our dataset</b> ("+ACT.toFixed(1)+"% failures): "+
        "a do-nothing model already looks ~"+(100-ACT).toFixed(0)+"% accurate.";
    } else if(share<ACT){
      note.textContent="Rarer failures → accuracy creeps toward 100%, yet the model is useless.";
    } else {
      note.textContent="More balanced classes → accuracy stops being so easy to fake.";
    }
  }
  p.addEventListener("input",upd); upd();
})();
</script>'''
    html = (tmpl.replace("__UID__", uid)
            .replace("__ACT__", repr(round(float(actual_failure_pct), 2))))
    display(HTML(html))


# --------------------------------------------------------------------- cost explorer
def cost_explorer(y_true, y_score):
    """Interactive cost / threshold explorer for the trained model."""
    uid = "ce_" + str(abs(hash((len(y_true), float(np.sum(y_score))))) % 10**8)
    grid = ('<table style="border-collapse:collapse;margin:10px 0;font-size:13px">'
            '<tr><td></td><td style="padding:4px 10px;color:#777">pred healthy</td>'
            '<td style="padding:4px 10px;color:#777">pred failure</td></tr>'
            '<tr><td style="color:#777">actual healthy</td>'
            '<td class="ce-tn" style="padding:8px 14px;background:#eef7ee;text-align:center;font-weight:700"></td>'
            '<td class="ce-fp2" style="padding:8px 14px;background:#fff3e8;text-align:center;font-weight:700"></td></tr>'
            '<tr><td style="color:#777">actual failure</td>'
            '<td class="ce-fn2" style="padding:8px 14px;background:#fdecec;text-align:center;font-weight:700"></td>'
            '<td class="ce-tp" style="padding:8px 14px;background:#e7f7ec;text-align:center;font-weight:700"></td></tr></table>')
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:620px;background:#fff;color:#2c2350}
#__UID__ .ce-head{font-weight:800;font-size:15px;margin-bottom:12px}
#__UID__ label{display:block;font-size:13px;margin:9px 0;color:#333}
#__UID__ input[type=range]{width:100%}
#__UID__ .ce-cost{margin-top:8px;font-size:15px;background:#f3f0ff;border-radius:8px;padding:10px 12px}
</style>
<div id="__UID__">
  <div class="ce-head">💸 What does this model cost the business?</div>
  <label>Cost of a FALSE ALARM (needless check): $<span class="ce-fpv">100</span>
    <input type="range" class="ce-fp" min="0" max="2000" step="50" value="100"></label>
  <label>Cost of a MISSED FAILURE (breakdown): $<span class="ce-fnv">5000</span>
    <input type="range" class="ce-fn" min="0" max="20000" step="500" value="5000"></label>
  <label>Decision threshold: <span class="ce-thv">0.50</span>
    <input type="range" class="ce-th" min="0.01" max="0.99" step="0.01" value="0.5"></label>
  __GRID__
  <div class="ce-cost"></div>
</div>
<script>
(function(){
  const Y=__Y__, S=__S__, root=document.getElementById("__UID__");
  const fp=root.querySelector(".ce-fp"), fn=root.querySelector(".ce-fn"), th=root.querySelector(".ce-th");
  function upd(){
    const t=+th.value; let TP=0,FP=0,FN=0,TN=0;
    for(let i=0;i<Y.length;i++){const p=S[i]>=t?1:0;
      if(p===1&&Y[i]===1)TP++; else if(p===1&&Y[i]===0)FP++;
      else if(p===0&&Y[i]===1)FN++; else TN++;}
    root.querySelector(".ce-fpv").textContent=fp.value;
    root.querySelector(".ce-fnv").textContent=fn.value;
    root.querySelector(".ce-thv").textContent=(+t).toFixed(2);
    root.querySelector(".ce-tp").textContent=TP; root.querySelector(".ce-fp2").textContent=FP;
    root.querySelector(".ce-fn2").textContent=FN; root.querySelector(".ce-tn").textContent=TN;
    const cost=FP*(+fp.value)+FN*(+fn.value), rec=(TP+FN)?TP/(TP+FN):0;
    const prec=(TP+FP)?TP/(TP+FP):NaN;
    const precTxt=isNaN(prec)?"—  <i>(flags nothing)</i>":(prec*100).toFixed(0)+"%";
    root.querySelector(".ce-cost").innerHTML="Total error cost: <b>$"+cost.toLocaleString()+
      "</b><br>Failures caught (recall): <b>"+(rec*100).toFixed(0)+"%</b>"+
      "<br>Of the flags, real failures (precision): <b>"+precTxt+"</b>";
  }
  [fp,fn,th].forEach(e=>e.addEventListener("input",upd)); upd();
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__GRID__", grid)
            .replace("__Y__", _json.dumps([int(v) for v in y_true]))
            .replace("__S__", _json.dumps([round(float(s), 4) for s in y_score])))
    display(HTML(html))
