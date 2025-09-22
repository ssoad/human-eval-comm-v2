#!/usr/bin/env python3
"""
Enhanced Flask Leaderboard for HumanEvalComm V2 Benchmark

Features:
- Auto-detects latest leaderboard files
- Interactive charts and visualizations
- Multiple filtering and sorting options
- Real-time data refresh
- Responsive design with modern UI
- Robust error handling
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from datetime import datetime

from config import Config
from data_manager import LeaderboardManager
from charts import ChartGenerator

app = Flask(__name__)
app.config.from_object(Config)

# Initialize manager and chart generator
manager = LeaderboardManager()
chart_generator = ChartGenerator(Config)

@app.route('/')
def index():
    """Main leaderboard page."""
    try:
        # Load data
        df = manager.load_leaderboard_data()
        files_info = manager.find_latest_files()
        
        if df is None:
            return render_template('error.html', 
                                 error="No leaderboard data found. Please run the V2 benchmark first.")
        
        # Get filtering parameters
        selected_model = request.args.get('model', '')
        sort_by = request.args.get('sort_by', 'V2 Score')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Apply filters
        filtered_df = df.copy()
        if selected_model:
            filtered_df = filtered_df[filtered_df['Model'] == selected_model]
        
        # Apply sorting
        if sort_by in filtered_df.columns:
            ascending = sort_order == 'asc'
            filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
        
        # Get summary stats
        stats = manager.get_summary_stats(filtered_df)
        
        # Get available models and columns
        models = sorted(df['Model'].unique()) if 'Model' in df else []
        columns = [col for col in df.columns if col != 'Model']
        
        return render_template('index.html',
                             df=filtered_df,
                             models=models,
                             columns=columns,
                             selected_model=selected_model,
                             sort_by=sort_by,
                             sort_order=sort_order,
                             stats=stats,
                             files_info=files_info)
                             
    except Exception as e:
        return render_template('error.html', error=f"Error loading leaderboard: {e}")

@app.route('/api/data')
def api_data():
    """API endpoint for leaderboard data."""
    try:
        df = manager.load_leaderboard_data()
        if df is None:
            return jsonify({'error': 'No data available'})
        
        return jsonify({
            'data': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'models': sorted(df['Model'].unique()) if 'Model' in df else [],
            'stats': manager.get_summary_stats(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/charts')
def api_charts():
    """API endpoint for chart data."""
    try:
        df = manager.load_leaderboard_data()
        if df is None:
            return jsonify({'error': 'No data available'})

        # Generate all chart data using ChartGenerator
        charts = chart_generator.generate_all_charts(df)

        return jsonify(charts)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/detailed')
def detailed():
    """Detailed results page."""
    try:
        results_data = manager.load_detailed_results()
        files_info = manager.find_latest_files()
        
        if results_data is None:
            return render_template('error.html', 
                                 error="No detailed results found. Please run the V2 benchmark first.")
        
        # Process results for display
        df_results = pd.DataFrame(results_data)
        
        # Get filtering parameters
        selected_model = request.args.get('model', '')
        selected_problem = request.args.get('problem', '')
        
        # Apply filters
        if selected_model:
            df_results = df_results[df_results['model_name'] == selected_model]
        if selected_problem:
            df_results = df_results[df_results['problem_id'] == selected_problem]
        
        # Get available options
        models = sorted(df_results['model_name'].unique()) if 'model_name' in df_results else []
        problems = sorted(df_results['problem_id'].unique()) if 'problem_id' in df_results else []
        
        # Load problems data for question display
        problems_data = {}
        try:
            # Try to load from HumanEvalComm_v2.jsonl
            problems_file = os.path.join(manager.base_dir, 'Benchmark',
                                         'HumanEvalComm_v2.jsonl')
            if os.path.exists(problems_file):
                problems_data = {}
                with open(problems_file, 'r') as f:
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
        
        # Format last updated timestamp
        last_updated = 'N/A'
        if (files_info.get('latest_results') and
                os.path.exists(files_info['latest_results'])):
            try:
                mtime = os.path.getmtime(files_info['latest_results'])
                last_updated = datetime.fromtimestamp(mtime).strftime(
                    '%Y-%m-%d %H:%M')
            except Exception:
                pass

        return render_template('detailed.html',
                               df=df_results,
                               models=models,
                               problems=problems,
                               selected_model=selected_model,
                               selected_problem=selected_problem,
                               files_info=files_info,
                               last_updated=last_updated,
                               problems_data=problems_data)

    except Exception as e:
        error_msg = f"Error loading detailed results: {e}"
        return render_template('error.html', error=error_msg)


@app.route('/refresh')
def refresh():
    """Refresh data cache."""
    try:
        manager.cache.clear()
        manager.cache_time.clear()
        return jsonify({'status': 'success', 'message': 'Cache refreshed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/files')
def files():
    """List available files."""
    try:
        files_info = manager.find_latest_files()
        return jsonify(files_info)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
