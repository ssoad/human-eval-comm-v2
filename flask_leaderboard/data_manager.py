"""
Data management module for HumanEvalComm Leaderboard
"""
import pandas as pd
import json
import os
import glob
from datetime import datetime
from config import Config


class LeaderboardManager:
    """Manages leaderboard data and operations."""

    def __init__(self):
        self.base_dir = Config.BASE_DIR
        self.data_dir = Config.DATA_DIR
        self.cache = {}
        self.cache_time = {}

    def find_latest_files(self):
        """Find the latest leaderboard and results files."""
        # Look for V2 leaderboard files
        leaderboard_pattern = os.path.join(self.data_dir,
                                           Config.LEADERBOARD_PATTERN)
        results_pattern = os.path.join(self.data_dir,
                                       Config.RESULTS_PATTERN)

        leaderboard_files = glob.glob(leaderboard_pattern)
        results_files = glob.glob(results_pattern)

        # Sort by modification time (newest first)
        leaderboard_files.sort(key=os.path.getmtime, reverse=True)
        results_files.sort(key=os.path.getmtime, reverse=True)

        return {
            'leaderboard_files': leaderboard_files,
            'results_files': results_files,
            'latest_leaderboard': (
                leaderboard_files[0] if leaderboard_files else None
            ),
            'latest_results': results_files[0] if results_files else None
        }

    def load_leaderboard_data(self, file_path=None):
        """Load leaderboard data with caching."""
        if file_path is None:
            files = self.find_latest_files()
            file_path = files['latest_leaderboard']

        if not file_path or not os.path.exists(file_path):
            return None

        # Check cache
        cache_key = f"leaderboard_{file_path}"
        file_mtime = os.path.getmtime(file_path)

        if (Config.CACHE_ENABLED and
                cache_key in self.cache and
                self.cache_time.get(cache_key, 0) >= file_mtime):
            return self.cache[cache_key]

        try:
            df = pd.read_csv(file_path)

            # Clean and process data with robust error handling
            for col in df.columns:
                if col == 'Model':
                    continue  # Skip model column

                # Convert all data columns to string first for cleaning
                df[col] = df[col].astype(str)

                # Clean malformed data like '100%100%' or '50%'
                if df[col].str.contains('%').any():
                    # Remove all % signs and extract first number
                    df[col] = df[col].str.replace('%', '', regex=False)
                    # Extract first number if there are multiple concatenated
                    df[col] = df[col].str.extract(
                        r'(\d+(?:\.\d+)?)', expand=False
                    )

                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Debug: print cleaned DataFrame info
            print(f"[DEBUG] Successfully loaded and cleaned {file_path}")
            print(f"[DEBUG] DataFrame shape: {df.shape}")
            print(f"[DEBUG] Columns: {df.columns.tolist()}")

            # Cache the data
            if Config.CACHE_ENABLED:
                self.cache[cache_key] = df
                self.cache_time[cache_key] = file_mtime

            return df

        except Exception as e:
            print(f"Error loading leaderboard data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_detailed_results(self, file_path=None):
        """Load detailed results data."""
        if file_path is None:
            files = self.find_latest_files()
            file_path = files['latest_results']

        if not file_path or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading results data: {e}")
            return None

    def load_problems_data(self):
        """Load problems data for question display."""
        problems_data = {}
        try:
            if os.path.exists(Config.PROBLEMS_FILE):
                with open(Config.PROBLEMS_FILE, 'r') as f:
                    for line in f:
                        problem = json.loads(line.strip())
                        problem_id = problem.get('name', '')
                        if problem_id:
                            # Use the appropriate prompt based on prompt_type
                            prompt = problem.get('prompt', '')
                            problems_data[problem_id] = {
                                'prompt': prompt,
                                'entry_point': problem.get('entry_point', ''),
                                'test_case': problem.get('test_case', [])
                            }
        except Exception as e:
            print(f"Warning: Could not load problems data: {e}")

        return problems_data

    def get_summary_stats(self, df):
        """Get summary statistics."""
        if df is None or df.empty:
            return {}

        stats = {
            'total_models': len(df),
            'avg_v2_score': df['V2 Score'].mean() if 'V2 Score' in df else 0,
            'avg_comm_rate': (df['Comm Rate'].mean()
                              if 'Comm Rate' in df else 0),
            'avg_pass_at_1': df['Pass@1'].mean() if 'Pass@1' in df else 0,
            'top_model': df.iloc[0]['Model'] if not df.empty else 'N/A',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return stats

    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.cache_time.clear()
