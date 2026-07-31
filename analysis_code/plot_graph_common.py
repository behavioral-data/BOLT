"""Shared rendering engine for the per-intent forest plots: one row per
model vs. a "Low / A / High" human reference band, built entirely from a
single significance CSV per aspect (see data/)."""
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sb
from scipy.stats import t as t_dist

# Keys as they appear in the CSV's 'model_name' column (as '{key}_all'),
# ordered bottom-to-top row in the plot.
MODEL_KEYS = ['llama2-13b', 'llama2-70b', 'gpt_35', 'gpt4']
MODEL_DISPLAY_ORDER = ['Llama2-13b', 'Llama2-70b', 'GPT-3.5', 'GPT-4']

POSITIVE_COLOR = '#85AABF'
NEGATIVE_COLOR = '#ED9265'
NOT_SIGNIFICANT_COLOR = '#e0e0e0'
OVERSHOOT_COLOR = '#decbe4'

# Row y-positions, bottom to top, matching MODEL_DISPLAY_ORDER.
MODEL_Y_POSITIONS = np.linspace(-1.75, 1.75, 4)


def apply_plot_style():
	matplotlib.rcParams['text.usetex'] = False
	matplotlib.rcParams['figure.autolayout'] = True

	sb.set_style('ticks', {'xtick.bottom': True, 'ytick.left': True, 'grid.color': '0.9'})
	sb.set_context(
		'talk',
		font_scale=0.9,
		rc={
			'lines.linewidth': 2,
			'text.usetex': False,
			'font.family': 'Lato',
			'font.sans-serif': ['Palatino'],
			'font.weight': 'bold',
			'font.size': 22,
			'xtick.major.size': 6,
			'xtick.major.width': 2,
			'ytick.left': True,
		},
	)


def form_list(first, middle, last):
	"""Build 9 evenly-spaced tick values: 4 between first/middle, the middle,
	then 4 between middle/last."""
	first_to_middle = np.linspace(first, middle, num=5)[1:4]
	middle_to_last = np.linspace(middle, last, num=5)[1:4]
	return [first] + first_to_middle.tolist() + [middle] + middle_to_last.tolist() + [last]


def process_numbers(numbers, middle, as_percent=True):
	"""Format tick values, blanking out any negative placeholder ticks."""
	suffix = '%' if as_percent else ''
	processed = []
	for number in numbers:
		if number < 0:
			processed.append('')
		else:
			processed.append(f'{round(number, 1)}{suffix}')
	return processed


def human_band_from_sig_df(df_sig):
	"""(human_avg, human_low, human_high) read from a sig_df slice's
	'human_avg'/'human_low'/'human_high' rows."""
	human_avg = df_sig[df_sig['model_name'] == 'human_avg']['avg'].values[0]
	human_low = df_sig[df_sig['model_name'] == 'human_low']['avg'].values[0]
	human_high = df_sig[df_sig['model_name'] == 'human_high']['avg'].values[0]
	return human_avg, human_low, human_high


def corrected_significance_from_sig_df(df_sig):
	"""Multiple-comparison-corrected significance flags: {model_key: bool},
	from columns 'model_name' (as '{key}_all') and 'pvalue' (already a
	corrected significant/not-significant boolean, not a raw p-value)."""
	return {
		key: bool(df_sig[df_sig['model_name'] == f'{key}_all']['pvalue'].values[0])
		for key in MODEL_KEYS
	}


def model_colors_from_sig_df(df_sig, human_avg, human_high, significant_by_key):
	"""Blue if above human avg, orange if below, grey if not significant.
	Divides by (human_high - human_avg), not just the sign of the
	difference, since that flips correctly when the axis is inverted."""
	colors = []
	high_diff = human_high - human_avg
	for key in MODEL_KEYS:
		model_avg = df_sig[df_sig['model_name'] == f'{key}_all']['avg'].values[0]
		relative_diff = (model_avg - human_avg) / high_diff if high_diff != 0 else (model_avg - human_avg)
		base_color = NEGATIVE_COLOR if relative_diff < 0 else POSITIVE_COLOR
		colors.append(base_color if significant_by_key[key] else NOT_SIGNIFICANT_COLOR)
	return colors


def model_stat_from_sig_row(sig_row, confidence=0.95):
	"""(mean, half_width) of a parametric CI from a sig_df row with 'avg',
	'std', 'size_for_model_name' columns."""
	mean = sig_row['avg']
	std = sig_row['std']
	n = sig_row['size_for_model_name']
	se = std / np.sqrt(n)
	crit = t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1)
	return mean, crit * se


def _draw_mixed_weight_line(fig, axs, y, label, value, fontsize):
	"""Centered 'label **value**' line at axes-fraction y (measures each
	segment since one Text object can't mix weights)."""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	ax_bbox = axs.get_window_extent(renderer=renderer)

	t_label = axs.text(0, y, label, fontsize=fontsize, fontweight='normal', transform=axs.transAxes, ha='left', va='bottom')
	fig.canvas.draw()
	b_label = t_label.get_window_extent(renderer=renderer)
	label_width_frac = (b_label.x1 - b_label.x0) / ax_bbox.width

	t_value = axs.text(0, y, ' ' + value, fontsize=fontsize, fontweight='bold', transform=axs.transAxes, ha='left', va='bottom')
	fig.canvas.draw()
	b_value = t_value.get_window_extent(renderer=renderer)
	value_width_frac = (b_value.x1 - b_value.x0) / ax_bbox.width

	total_frac = label_width_frac + value_width_frac
	start_x = 0.5 - total_frac / 2

	t_label.set_position((start_x, y))
	t_value.set_position((start_x + label_width_frac, y))


def _resolve_text_collisions(fig, axs, texts, min_gap_px=10):
	"""Spread same-row Text objects apart (in data space) if they overlap on
	screen, e.g. "Low"/"A"/"High" when the human band is narrow. One
	left-to-right pass ordered by box center (not left edge, since "Low"/
	"High" are wider than "A") so texts can't cross past each other."""
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	order = sorted(texts, key=lambda t: (lambda b: (b.x0 + b.x1) / 2)(t.get_window_extent(renderer=renderer)))
	boxes = [t.get_window_extent(renderer=renderer) for t in order]
	widths = [b.x1 - b.x0 for b in boxes]
	centers = [(b.x0 + b.x1) / 2 for b in boxes]

	new_centers = [centers[0]]
	for i in range(1, len(order)):
		min_spacing = (widths[i - 1] + widths[i]) / 2 + min_gap_px
		new_centers.append(max(centers[i], new_centers[-1] + min_spacing))

	group_shift = (sum(centers) - sum(new_centers)) / len(centers)
	new_centers = [c + group_shift for c in new_centers]

	for t, old_center_px, new_center_px in zip(order, centers, new_centers):
		if abs(new_center_px - old_center_px) < 0.01:
			continue
		x_data_old = axs.transData.inverted().transform((old_center_px, 0))[0]
		x_data_new = axs.transData.inverted().transform((new_center_px, 0))[0]
		pos = t.get_position()
		t.set_position((pos[0] + (x_data_new - x_data_old), pos[1]))


def draw_stacked_title(fig, axs, title_lines, fontsize=15, line_spacing=0.16, top=1.28):
	"""title_lines: list of (label, value) pairs, each rendered as one
	centered 'label **value**' line, stacked above the axes (topmost
	first)."""
	for i, (label, value) in enumerate(title_lines):
		_draw_mixed_weight_line(fig, axs, top - i * line_spacing, label, value, fontsize)


def render_forest_plot_from_summary(
	model_stats,
	human_avg,
	human_low,
	human_high,
	title,
	output_path,
	as_percent=True,
	overshoot_value=None,
	text_shift=(0.0, 0.0, 0.0),
	line_shift=0.0,
	arrow_head_shift=0.0,
	arrow_tail_shift=0.0,
	title_lines=None,
):
	"""Render one horizontal point-plot (one row per model) with a human
	Low/Avg/High reference band, and save it to output_path.

	model_stats: list of (mean, half_width, color) in MODEL_DISPLAY_ORDER.
	*_shift params: fraction of the axis span, unused by any current caller
		(Low/A/High auto-separate) -- kept as an escape hatch.
	title_lines: optional [(label, value), ...] stacked above the axes
		instead of the plain bold `title`.
	"""
	fig = plt.figure(figsize=(10, 3))
	axs = plt.gca()

	for (mean, half_width, color), model_name, y in zip(model_stats, MODEL_DISPLAY_ORDER, MODEL_Y_POSITIONS):
		axs.errorbar(mean, y, xerr=half_width, fmt='o', color=color, ecolor=color, elinewidth=2, markersize=8, zorder=2)
		plt.text(mean, y + 0.4, model_name, color='black', ha='center', fontsize=13)

	plt.ylim(min(MODEL_Y_POSITIONS) - 0.7, max(MODEL_Y_POSITIONS) + 0.9)
	ymin, ymax = axs.get_ylim()
	axs.xaxis.grid(False)
	plt.yticks([])

	xmin, xmax = plt.xlim()
	xmin_diff = human_avg - xmin
	xmax_diff = xmax - human_avg
	high_diff = abs(human_high - human_avg)

	relative_xmin_diff = int(np.ceil(abs(xmin_diff / high_diff))) if high_diff > 0 else 10
	relative_xmax_diff = int(np.ceil(abs(xmax_diff / high_diff))) if high_diff > 0 else 10
	relative_absmax = max(relative_xmin_diff, relative_xmax_diff)
	if relative_absmax <= 4:
		relative_absmax = 4

	xmax = (relative_absmax * high_diff) + human_avg
	xmin = human_avg - (relative_absmax * high_diff)
	axs.set_xlim(xmin, xmax)
	axis_span = abs(xmax - xmin)

	middle = human_avg
	# +high_diff/2, not a fixed value: keeps ticks from overshooting xmax
	# and silently expanding (and de-centering) the axis.
	axs.set_xticks(np.arange(xmin, xmax + high_diff / 2, high_diff))
	xticks = plt.xticks()[0]
	xticks_new = xticks
	plt.xticks(xticks, xticks_new)
	plt.xticks(fontsize=14)
	plt.xticks(rotation=0)
	axs.xaxis.set_major_locator(ticker.LinearLocator(9))

	first = float(xticks_new[0])
	last = float(xticks_new[-1])

	selected_xticks = form_list(first, middle, last)
	selected_xticks = process_numbers(selected_xticks, middle, as_percent=as_percent)
	axs.set_xticklabels(selected_xticks)
	text_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.2

	if human_low > human_high:
		plt.gca().invert_xaxis()

	line_shift_abs = line_shift * axis_span
	line_avg = human_avg + line_shift_abs
	line_low = human_low + line_shift_abs
	line_high = human_high + line_shift_abs
	plt.axvline(x=line_avg, color='black', linestyle='-', linewidth=0.9, zorder=0)
	plt.axvline(x=line_low, color='#9e9e9e', linestyle='-', linewidth=0.9, zorder=0)
	plt.axvline(x=line_high, color='#9e9e9e', linestyle='-', linewidth=0.9, zorder=0)

	if overshoot_value is not None:
		plt.axvline(x=overshoot_value, color=OVERSHOOT_COLOR, linestyle='--', linewidth=0.9, zorder=0)

	text_low_shift, text_avg_shift, text_high_shift = (s * axis_span for s in text_shift)
	low_text = plt.text(human_low + text_low_shift, plt.ylim()[0] - text_offset, 'Low',
			  ha='center', va='top', color='black', fontweight='normal', fontsize=14)
	avg_text = plt.text(human_avg + text_avg_shift, plt.ylim()[0] - text_offset, 'A',
			  ha='center', va='top', color='black', fontweight='bold', fontsize=14)
	high_text = plt.text(human_high + text_high_shift, plt.ylim()[0] - text_offset, 'High',
			  ha='center', va='top', color='black', fontweight='normal', fontsize=14)
	_resolve_text_collisions(fig, axs, [low_text, avg_text, high_text])

	arrow_xy = xmin + arrow_head_shift * axis_span
	arrow_xytext = xmax + arrow_tail_shift * axis_span
	if first < last:
		axs.annotate('', xy=(arrow_xy, ymin), xytext=(arrow_xytext, ymin), arrowprops=dict(arrowstyle='<|-', lw=1.5, color='black'))
	else:
		axs.annotate('', xy=(arrow_xy, ymin), xytext=(arrow_xytext, ymin), arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'))

	tick_positions = axs.get_xticks()
	for pos, label in zip(tick_positions, axs.get_xticklabels()):
		if label.get_text() == '':
			label.set_visible(False)
			idx = tick_positions.tolist().index(pos)
			axs.xaxis.get_major_ticks()[idx].tick1line.set_visible(False)
			axs.xaxis.get_major_ticks()[idx].tick2line.set_visible(False)

	plt.ylabel('', fontsize=15)
	plt.xlabel('', fontsize=15)
	if title_lines:
		# tight_layout() before subplots_adjust(): headroom for the stacked
		# title must survive, and bottom=0.2545 matches the else-branch's
		# effective axes height so Low/A/High spacing stays consistent.
		fig.tight_layout()
		plt.subplots_adjust(left=0.1, right=0.9, top=0.72, bottom=0.2545)
		draw_stacked_title(fig, axs, title_lines)
	else:
		plt.title(title, fontsize=20, fontweight='bold')
		plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
		fig.tight_layout()

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fig.savefig(output_path, dpi=300)
	plt.close(fig)
