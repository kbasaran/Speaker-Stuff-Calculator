"""Generation of the standalone HTML report.

The report is one self-contained file: the results summary, the input-form
values that the summary does not already carry, and the graphs as base64 PNGs.
Nothing is linked from outside, so the document can be mailed or archived on
its own.

The graphs are drawn by a second, never-shown MatplotlibWidget rather than by
the one on screen. That keeps every styling decision (log-frequency ticks,
grids, the per-plot y-limit policies, legend ordering) in one place instead of
being re-implemented here, and it leaves the user's current graph selection
untouched while the report is built.
"""

import base64
import html
import io
import json
import logging
import math
import re
import time
from itertools import groupby
from pathlib import Path
from string import Template

from PySide6 import QtWidgets as qtw

from generictools import signal_tools
from generictools.graphing_widget import MatplotlibWidget

from config.app_config import APP_DEFINITIONS, singleton_settings
from gui import labels
from gui.plot_builders import PLOT_BUILDERS, apply_spec, graph_is_available
from utils.paths import get_main_dir

logger = logging.getLogger(__name__)
app_settings = singleton_settings()

FILE_FILTER = "HTML documents (*.html)"
FILE_SUFFIX = ".html"

# Relative to the installation folder; bundled with the rest of data/ by the
# packaging script.
TEMPLATE_FILE = "data/report_template.html"

# Source voltage the reference set of graphs is drawn at, alongside the graphs
# for whatever excitation the user currently has set up.
REFERENCE_VOLTAGE = 2.83

# Images are rendered at this multiple of the application's screen dpi and then
# displayed at the application's own pixel size, so they stay sharp on
# high-resolution displays and in print.
IMAGE_SCALE = 2

# graph_data_choice ids whose curves do not depend on the drive level, so they
# are reported once instead of once per excitation. Impedance is a property of
# the electrical network and the phases are ratios, so neither moves with the
# voltage.
VOLTAGE_INDEPENDENT_GRAPHS = (1, 6)

# Input-form values the report leaves out, so that the input section stays
# complementary to the results summary rather than repeating it, and carries
# only what says something about the design. The entries naming a place in the
# summary must be kept in sync with SpeakerSystem.get_summary and the summaries
# it calls into (SpeakerDriver, Motor, Coil).
INPUTS_NOT_REPORTED = frozenset({
    # already in the results summary, under the heading named
    "Le",           # "L_e" under Speaker unit
    "Xpeak",        # "X_peak" under Displacements
    "Bl_p2", "Bl_p3",    # "Bl"
    "Re_p2", "Re_p3",    # "R_e"
    "Mmd_p2",       # "M_md"
    "Mms_p3",       # "M_ms"
    "Qp",           # "Q_p" under Bass reflex
    "coil_options",  # spelled out under Windings

    # nothing to say about the design
    "excitation_type",   # every graph it applies to states the excitation itself
    "excitation_value",
    "motor_spec_type",   # only records which way the numbers were entered
    })

# Inputs whose value names the kind of part first and the way it is specified
# after a " - "; only the part is of interest, e.g. "PR - Define mass and damping
# ratio" is reported as "PR".
INPUTS_VALUE_BEFORE_DASH = frozenset({
    "resonator_spec_type",
    })

# Abbreviations for the units the tooltips spell out in words. Units the tooltips
# already give in short form (cm², mm, mH, ohm) pass through untouched, as does a
# parenthesised symbol (see _UNIT_SYMBOL_PATTERN).
UNIT_ABBREVIATIONS = {
    "Hertz": "Hz",
    "gram": "g",
    "liter": "l",
    "millimeter": "mm",
    "seconds": "s",
    "Tesla": "T",
    "Tesla meter": "Tm",
    "kilograms per second": "kg/s",
    "Newtons per millimeter": "N/mm",
    }

# Labels the report uses in place of what the form shows, taking precedence over
# the form's own row label. These inputs are not laid out as labelled QFormLayout
# rows and so have no label widget to read: the enclosure/parent-body radio groups
# and the combo box heading a stacked page.
WIDGET_LABELS = {
    "resonator_spec_type": "Resonator",
    "enclosure_type": "Enclosure type",
    "parent_body": "Parent body",
    }

# The tooltips state the unit of an input on a line of their own, e.g.
# "Unit is millimeter." -- that wording is the application's only record of an
# input's unit, so the report reads it from there.
_UNIT_PATTERN = re.compile(r"^Unit is (.+?)\.$", re.MULTILINE)
# A parenthesised symbol at the end of a unit, e.g. "micrometer (μm)".
_UNIT_SYMBOL_PATTERN = re.compile(r"\(([^)]+)\)$")


# ---- Input form ------------------------------------------------------------

def _unit_from_tooltip(tooltip: str) -> str:
    """The unit an input is entered in, abbreviated. '' when it has none.

    The tooltips are written for the form, where there is room to spell a unit
    out; the report sets the unit beside the number instead, so it wants the
    short form.
    """
    match = _UNIT_PATTERN.search(tooltip or "")
    if match is None:
        return ""

    unit = match.group(1).strip()

    # Prefer the symbol when the tooltip spells the unit out and then gives one.
    symbol = _UNIT_SYMBOL_PATTERN.search(unit)
    if symbol:
        return symbol.group(1)

    return UNIT_ABBREVIATIONS.get(unit, unit)


def _is_on_inactive_page(widget) -> bool:
    """Whether the widget sits on a QStackedWidget page that is not the shown one.

    The inputs of the motor and resonator definitions live on stacked pages, and
    the pages that are not on show stay *enabled* -- Qt only hides them -- so
    isEnabled() alone would report a motor definition the model was not built
    from. isVisible() cannot stand in for it either: it is False for every tab
    that is not the current one, and the report covers all four tabs. Hence this
    walk, which skips the stack a QTabWidget drives its own pages with.
    """
    child = widget
    parent = widget.parentWidget()
    while parent is not None:
        if (isinstance(parent, qtw.QStackedWidget)
                and not isinstance(parent.parentWidget(), qtw.QTabWidget)
                and parent.currentWidget() is not child):
            return True
        child, parent = parent, parent.parentWidget()
    return False


def _label_for(widget) -> str | None:
    "The text of the form-row label of a widget, or None when it has no label."
    parent = widget.parentWidget()
    while parent is not None:
        layout = parent.layout()
        if isinstance(layout, qtw.QFormLayout):
            label = layout.labelForField(widget)
            if label is not None:
                return label.text()
        parent = parent.parentWidget()
    return None


def _display_value(widget) -> str:
    "What the widget shows the user, as text. '' for widgets that hold no value."
    if isinstance(widget, qtw.QButtonGroup):
        button = widget.checkedButton()
        return button.text() if button is not None else ""
    if isinstance(widget, qtw.QComboBox):
        return widget.currentText()
    if isinstance(widget, qtw.QAbstractSpinBox):
        return widget.cleanText()
    if isinstance(widget, qtw.QCheckBox):
        return "Yes" if widget.isChecked() else "No"
    if isinstance(widget, qtw.QLineEdit):
        return widget.text()
    return ""


def collect_extra_inputs(input_form) -> list[tuple[str, list[dict]]]:
    """The active input values worth reporting beside the results summary.

    Left out are the values the summary already carries and the handful that say
    nothing about the design (see INPUTS_NOT_REPORTED).

    Returns one (tab name, rows) pair per tab that has anything to show, with the
    rows in the order the form lays them out. Only inputs that take part in the
    current model are listed: a disabled input has been switched off by the form
    logic, and one on a hidden stacked page belongs to a definition that was not
    used.
    """
    sections = []

    for i_tab in range(input_form.count()):
        form = input_form.widget(i_tab)
        rows = []

        for name, widget in form.interactable_widgets.items():
            if name in INPUTS_NOT_REPORTED:
                continue
            if isinstance(widget, qtw.QAbstractButton) and not isinstance(widget, qtw.QCheckBox):
                continue  # a push button holds no value

            # A button group is not a widget; judge it by the button that is on.
            probe = widget.checkedButton() if isinstance(widget, qtw.QButtonGroup) else widget
            if probe is None or not probe.isEnabled() or _is_on_inactive_page(probe):
                continue

            value = _display_value(widget)
            if not value:
                continue
            if name in INPUTS_VALUE_BEFORE_DASH:
                value = value.split(" - ")[0].strip()

            rows.append({"label": WIDGET_LABELS.get(name) or _label_for(probe) or name,
                         "value": value,
                         "unit": _unit_from_tooltip(probe.toolTip()),
                         "tooltip": probe.toolTip(),
                         })

        if rows:
            sections.append((input_form.tabText(i_tab), rows))

    return sections


# ---- Graphs ----------------------------------------------------------------

def _excitations_for(graph_id: int, V_source: float) -> list[float]:
    "The source voltages one graph is reported at."
    if graph_id in VOLTAGE_INDEPENDENT_GRAPHS:
        return [V_source]

    if math.isclose(V_source, REFERENCE_VOLTAGE, rel_tol=1e-9):
        return [REFERENCE_VOLTAGE]  # the current input already is the reference
    return [REFERENCE_VOLTAGE, V_source]


def render_graphs(graph_widget, spk_sys, V_source: float) -> list[dict]:
    """Render every graph the model supports, at the reference and current input.

    The graphs come out in choice-button order, and the ones belonging to the
    same choice are adjacent, which is what lets the document lay each pair out
    side by side.

    'graph_widget' is only read from, for the size the figures are drawn at;
    the drawing happens on a widget of this function's own.
    """
    freqs = signal_tools.generate_log_spaced_freq_list(app_settings.get_value("f_min"),
                                                       app_settings.get_value("f_max"),
                                                       app_settings.get_value("calc_ppo"),
                                                       )

    on_screen_figure = graph_widget.canvas.figure
    size_inches = tuple(on_screen_figure.get_size_inches())
    dpi = float(on_screen_figure.dpi)

    offscreen = MatplotlibWidget(layout_engine="tight")
    offscreen.canvas.figure.set_size_inches(*size_inches)
    offscreen.canvas.figure.set_dpi(dpi)

    graphs = []
    try:
        for graph_id, builder in sorted(PLOT_BUILDERS.items()):
            # The report holds exactly the graphs the application would let the
            # user look at, instead of pages saying the model has no such part.
            if not graph_is_available(graph_id, spk_sys):
                continue

            for V_graph in _excitations_for(graph_id, V_source):
                V_spk, W_spk = labels.excitation_at_speaker(spk_sys, V_graph)
                spec = builder(spk_sys, freqs, V_graph, V_spk, W_spk)
                apply_spec(offscreen, spec, freqs,
                           app_settings.get_value("f_min"),
                           app_settings.get_value("f_max"),
                           )
                graphs.append({"id": graph_id,
                               "alt": spec.title.replace("\n", " - "),
                               "png": _figure_to_png(offscreen.canvas.figure, dpi),
                               "width": round(size_inches[0] * dpi),
                               "height": round(size_inches[1] * dpi),
                               })
    finally:
        offscreen.deleteLater()

    return graphs


def _figure_to_png(figure, dpi: float) -> bytes:
    "The figure as PNG bytes, rendered at IMAGE_SCALE times its screen resolution."
    buffer = io.BytesIO()
    # Only the resolution is raised; the size in inches -- and with it every font
    # size and line width, which are set in points -- is left alone, so the image
    # is the same picture with more pixels in it.
    figure.savefig(buffer, format="png", dpi=dpi * IMAGE_SCALE)
    return buffer.getvalue()


# ---- Document --------------------------------------------------------------

def _inputs_html(sections: list[tuple[str, list[dict]]]) -> str:
    """The input sections as one table per tab. Labels are already HTML fragments.

    The unit sits beside the number in the value cell rather than in a column of
    its own, so the table fits in half a page.
    """
    if not sections:
        return "<p>-</p>"

    parts = []
    for section_name, rows in sections:
        parts.append(f"<h3>{html.escape(section_name)}</h3>")
        parts.append('<table class="inputs">')
        for row in rows:
            title = f' title="{html.escape(row["tooltip"])}"' if row["tooltip"] else ""
            unit = (f' <span class="unit">{html.escape(row["unit"])}</span>'
                    if row["unit"] else "")
            parts.append(f'<tr{title}>'
                         f'<td class="label">{row["label"]}</td>'
                         f'<td class="value">{html.escape(row["value"])}{unit}</td>'
                         '</tr>')
        parts.append("</table>")

    return "\n".join(parts)


def _graph_rows(graphs: list[dict]) -> list[list[dict]]:
    """The graphs arranged into rows of at most two, in graph-choice order.

    A choice reported at both drive levels keeps a row to itself, so its two
    graphs always sit side by side to be compared. A choice with a single graph
    -- the voltage-independent ones, and every choice when the current input
    already is the reference voltage -- shares its row with the next single one
    instead of leaving half the page empty down the whole section. Only
    neighbours are paired up, so the order of the graph choices is kept.
    """
    rows = []
    for _, group in groupby(graphs, key=lambda graph: graph["id"]):
        group = list(group)
        if len(group) == 1 and rows and len(rows[-1]) == 1:
            rows[-1].append(group[0])
        else:
            rows.append(group)
    return rows


def _graphs_html(graphs: list[dict]) -> str:
    """The graphs as base64 PNGs, laid out two to a row.

    No heading is written: every figure carries its own title and excitation
    line, drawn into the image.
    """
    if not graphs:
        return "<p>-</p>"

    parts = []
    for row in _graph_rows(graphs):
        parts.append('<div class="graph-row">')
        for graph in row:
            encoded = base64.b64encode(graph["png"]).decode("ascii")
            parts.append('<figure class="graph">'
                         f'<img src="data:image/png;base64,{encoded}"'
                         f' width="{graph["width"]}" height="{graph["height"]}"'
                         f' alt="{html.escape(graph["alt"])}">'
                         '</figure>')
        parts.append("</div>")

    return "\n".join(parts)


def _state_json(state: dict) -> str:
    """The session state, for a report to stay traceable to the model behind it.

    Escaped so that no value can close the script element it is carried in.
    """
    state = {**state, "application_data": APP_DEFINITIONS}
    return json.dumps(state, indent=4).replace("</", "<\\/")


def _meta_line() -> str:
    return (f"Generated {time.strftime('%Y-%m-%d %H:%M')}"
            f" &middot; {APP_DEFINITIONS['app_name']} {APP_DEFINITIONS['version']}"
            f" &middot; calculated over {app_settings.get_value('f_min'):g}"
            f"-{app_settings.get_value('f_max'):g} Hz"
            f" at {app_settings.get_value('calc_ppo'):g} points per octave"
            )


def build_html(title: str, description: str, results_html: str,
               input_sections: list, graphs: list, subtitle: str, state: dict) -> str:
    "Fill the report template. All caller-supplied text is escaped here."
    template = Template(get_main_dir().joinpath(TEMPLATE_FILE).read_text(encoding="utf-8"))

    return template.substitute(
        app_name=html.escape(APP_DEFINITIONS["app_name"]),
        app_version=html.escape(APP_DEFINITIONS["version"]),
        title=html.escape(title.strip() or "Untitled"),
        subtitle=html.escape(subtitle),
        description=html.escape(description),
        results=results_html,  # already HTML, straight from the model's summary
        inputs=_inputs_html(input_sections),
        graphs=_graphs_html(graphs),
        meta=_meta_line(),
        state_json=_state_json(state),
        )


# ---- File IO ---------------------------------------------------------------

def _default_file_name(title: str) -> str:
    "A file name suggestion made from the design's title."
    stem = re.sub(r"[^\w \-.]", "", title).strip() or "report"
    return stem + FILE_SUFFIX


def prompt_report_path(parent, start_dir: str, title: str) -> Path | None:
    "Show the save dialog and return the chosen path, or None if canceled."
    path_unverified = qtw.QFileDialog.getSaveFileName(parent,
                                                      caption="Save report to a file..",
                                                      dir=str(Path(start_dir) / _default_file_name(title)),
                                                      filter=FILE_FILTER,
                                                      )
    file_raw = path_unverified[0]
    if not file_raw:
        return None  # nothing selected, so pick file is canceled

    file = Path(file_raw)
    if file.suffix.lower() != FILE_SUFFIX:
        # the filter does not append the extension itself; mirrors session_io
        file = file.with_suffix(FILE_SUFFIX)

    return file


def write_report(file: Path, document: str) -> None:
    "Write the report document to disk."
    logger.info(f"Writing report '{file.name}'")
    file.write_text(document, encoding="utf-8")
