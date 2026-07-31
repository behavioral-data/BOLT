"""Forest plot: temporal order (position in the conversation) of each
intent, per model, vs. the human reference band. One PNG per intent.

Input: data/all_order.csv only.
Output: output/temporal_order/{intent}.png

Omits the original pipeline's dashed "overshoot" threshold line (needs a
separate CSV not included in data/) -- see the README.
"""
import os

import pandas as pd

import plot_graph_common as pgc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIG_CSV_PATH = os.path.join(BASE_DIR, 'data/all_order.csv')
SAVE_DIR_GRAPH = os.path.join(BASE_DIR, 'output/temporal_order')


def main():
	pgc.apply_plot_style()
	os.makedirs(SAVE_DIR_GRAPH, exist_ok=True)

	sig_df = pd.read_csv(SIG_CSV_PATH)
	intent_list = list(sig_df['intent'].unique())

	for intent in intent_list:
		df_sig = sig_df[sig_df['intent'] == intent]

		human_avg, human_low, human_high = pgc.human_band_from_sig_df(df_sig)
		significant = pgc.corrected_significance_from_sig_df(df_sig)
		colors = pgc.model_colors_from_sig_df(df_sig, human_avg, human_high, significant)

		model_stats = []
		for key, color in zip(pgc.MODEL_KEYS, colors):
			sig_row = df_sig[df_sig['model_name'] == f'{key}_all'].iloc[0]
			mean, half_width = pgc.model_stat_from_sig_row(sig_row)
			model_stats.append((mean, half_width, color))

		pgc.render_forest_plot_from_summary(
			model_stats, human_avg, human_low, human_high, intent,
			output_path=f'{SAVE_DIR_GRAPH}/{intent}.png',
			as_percent=False,
		)

	print(f'Wrote {len(intent_list)} charts to {SAVE_DIR_GRAPH}/')


if __name__ == '__main__':
	main()
