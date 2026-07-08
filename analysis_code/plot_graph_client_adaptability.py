"""Forest plot: how often each model responds to a given client intent with
a given therapist intent, vs. the human reference band. One PNG per (client
intent, therapist intent) pair.

Input: data/all_dist_clientvst.csv only.
Output: output/client_adaptability/{client_intent}_{therapist_intent}.png

Omits the original pipeline's dashed "overshoot" threshold line (needs a
separate CSV not included in data/) -- see the README.
"""
import os

import pandas as pd

import plot_graph_common as pgc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIG_CSV_PATH = os.path.join(BASE_DIR, 'data/all_dist_clientvst.csv')
SAVE_DIR_GRAPH = os.path.join(BASE_DIR, 'output/client_adaptability')


def main():
	pgc.apply_plot_style()
	os.makedirs(SAVE_DIR_GRAPH, exist_ok=True)

	sig_df = pd.read_csv(SIG_CSV_PATH)
	pairs = sig_df[['intent_therapist', 'intent_client']].drop_duplicates()

	for intent_therapist, intent_client in pairs.itertuples(index=False):
		df_sig = sig_df[(sig_df['intent_therapist'] == intent_therapist) & (sig_df['intent_client'] == intent_client)]

		human_avg, human_low, human_high = pgc.human_band_from_sig_df(df_sig)
		significant = pgc.corrected_significance_from_sig_df(df_sig)
		colors = pgc.model_colors_from_sig_df(df_sig, human_avg, human_high, significant)

		model_stats = []
		for key, color in zip(pgc.MODEL_KEYS, colors):
			sig_row = df_sig[df_sig['model_name'] == f'{key}_all'].iloc[0]
			mean, half_width = pgc.model_stat_from_sig_row(sig_row)
			model_stats.append((mean, half_width, color))

		pgc.render_forest_plot_from_summary(
			model_stats, human_avg, human_low, human_high, intent_therapist,
			output_path=f'{SAVE_DIR_GRAPH}/{intent_client}_{intent_therapist}.png',
			as_percent=True,
			title_lines=[('Client behavior:', intent_client), ('Therapist behavior in response:', intent_therapist)],
		)

	print(f'Wrote {len(pairs)} charts to {SAVE_DIR_GRAPH}/')


if __name__ == '__main__':
	main()
