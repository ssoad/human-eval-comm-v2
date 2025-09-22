"""
Chart generation module for HumanEvalComm Leaderboard
"""


class ChartGenerator:
    """Generates chart data for visualizations."""

    def __init__(self, config):
        self.config = config

    def generate_all_charts(self, df):
        """Generate all chart data from dataframe."""
        if df is None:
            return self._get_empty_charts()

        charts = {
            'v2_scores': self._generate_v2_scores_chart(df),
            'communication': self._generate_communication_chart(df),
            'performance_radar': self._generate_performance_radar(df),
            'trustworthiness': self._generate_trustworthiness_chart(df),
            'heatmap': self._generate_heatmap(df)
        }

        self._debug_chart_summary(charts)
        return charts

    def _get_empty_charts(self):
        """Return empty chart structure."""
        return {
            'v2_scores': {'x': [], 'y': []},
            'communication': {
                'models': [], 'comm_rate': [], 'good_q_rate': []
            },
            'performance_radar': {
                'models': [], 'metrics': [], 'values': []
            },
            'trustworthiness': {'models': []},
            'heatmap': {'x': [], 'y': [], 'z': []}
        }

    def _generate_v2_scores_chart(self, df):
        """Generate V2 scores comparison chart."""
        if 'V2 Score' not in df.columns:
            return {'x': [], 'y': []}

        return {
            'x': df['Model'].tolist(),
            'y': df['V2 Score'].tolist()
        }

    def _generate_communication_chart(self, df):
        """Generate communication metrics chart."""
        if ('Comm Rate' not in df.columns or
                'Good Q Rate' not in df.columns):
            return {'models': [], 'comm_rate': [], 'good_q_rate': []}

        return {
            'models': df['Model'].tolist(),
            'comm_rate': df['Comm Rate'].tolist(),
            'good_q_rate': df['Good Q Rate'].tolist()
        }

    def _generate_performance_radar(self, df):
        """Generate performance radar chart."""
        performance_cols = [
            'Pass@1', 'Test Pass', 'Readability',
            'Security', 'Efficiency', 'Reliability'
        ]
        available_cols = [
            col for col in performance_cols if col in df.columns
        ]

        if not available_cols:
            return {'models': [], 'metrics': [], 'values': []}

        models = df['Model'].tolist()
        values = []

        for _, row in df.iterrows():
            values.append([row[col] for col in available_cols])

        return {
            'models': models,
            'metrics': available_cols,
            'values': values
        }

    def _generate_trustworthiness_chart(self, df):
        """Generate trustworthiness metrics chart."""
        trust_metrics = [
            'Readability', 'Security', 'Efficiency', 'Reliability'
        ]
        available_metrics = [
            m for m in trust_metrics if m in df.columns
        ]

        if not available_metrics:
            return {'models': []}

        trust = {'models': df['Model'].tolist()}
        for metric in available_metrics:
            trust[metric] = df[metric].tolist()

        return trust

    def _generate_heatmap(self, df):
        """Generate performance heatmap."""
        performance_cols = [
            'Pass@1', 'Test Pass', 'Readability',
            'Security', 'Efficiency', 'Reliability'
        ]
        available_cols = [
            col for col in performance_cols if col in df.columns
        ]

        if not available_cols:
            return {'x': [], 'y': [], 'z': []}

        heatmap_z = []
        for _, row in df.iterrows():
            heatmap_z.append([row[col] for col in available_cols])

        return {
            'x': available_cols,
            'y': df['Model'].tolist(),
            'z': heatmap_z
        }

    def _debug_chart_summary(self, charts):
        """Print debug summary of generated charts."""
        print('[DEBUG] Charts generated:')
        v2_count = len(charts.get('v2_scores', {}).get('x', []))
        comm_count = len(charts.get('communication', {}).get('models', []))
        radar_count = len(charts.get('performance_radar', {})
                          .get('models', []))
        trust_count = len(charts.get('trustworthiness', {})
                          .get('models', []))
        heatmap_count = len(charts.get('heatmap', {}).get('y', []))

        print(f'  - V2 scores: {v2_count} models')
        print(f'  - Communication: {comm_count} models')
        print(f'  - Performance radar: {radar_count} models')
        print(f'  - Trustworthiness: {trust_count} models')
        print(f'  - Heatmap: {heatmap_count} models')
