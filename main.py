from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your secret key'  # change this to a real secret key

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        df1 = pd.read_csv(file1.stream, sep=';')
        df2 = pd.read_csv(file2.stream, sep=';')

        result1 = analyse_df(df1)
        result2 = analyse_df(df2)

        comparison = compare_results(result1, result2)

        # Convert results to serializable format before saving in session
        session['comparison'] = comparison.to_dict()

        return redirect(url_for('results'))  # Redirect to the results route

    return render_template('upload.html')

@app.route('/results', methods=['GET'])
def results():
    # Retrieve the result from session
    comparison = session.get('comparison', None)
    return render_template('results.html', comparison=comparison)


def analyse_df(df):
    result = {}
    for col in df.columns:
        # Detect column type
        col_type = df[col].dtypes

        if np.issubdtype(col_type, np.number):
            # For numeric columns
            result[col] = {
                'sum': df[col].sum(),
                'null_values': df[col].isnull().sum(),
                'total_values': df[col].count()
            }
        elif np.issubdtype(col_type, np.bool_):
            # For boolean columns
            result[col] = {
                'true_values': df[col].sum(),
                'false_values': len(df[col]) - df[col].sum(),
                'null_values': df[col].isnull().sum(),
                'total_values': df[col].count()
            }
        elif col_type == 'object':
            # For string columns
            result[col] = {
                'total_values': df[col].replace('', np.nan).count(),
                'distinct_values': len(df[col].unique()),
                'null_values': df[col].isnull().sum()
            }
        elif np.issubdtype(col_type, np.datetime64):
            # For date columns
            result[col] = {
                'sum': pd.to_datetime(df[col]).astype(int).sum(),
                'null_values': df[col].isnull().sum(),
                'total_values': df[col].count()
            }

    # Convert result to DataFrame for serializability
    return pd.DataFrame(result)


def compare_results(result1, result2):
    comparison = {}
    for col in result1:
        if col in result2:  # check if column exists in both dataframes
            comparison[col] = {
                'total_values_file1': result1[col]['total_values'],
                'total_values_file2': result2[col]['total_values'],
                'total_values_same': result1[col]['total_values'] == result2[col]['total_values'],
                'null_values_file1': result1[col]['null_values'],
                'null_values_file2': result2[col]['null_values'],
                'null_values_same': result1[col]['null_values'] == result2[col]['null_values']
            }
            if 'distinct_values' in result1[col] and 'distinct_values' in result2[col]:
                comparison[col].update({
                    'distinct_values_file1': result1[col]['distinct_values'],
                    'distinct_values_file2': result2[col]['distinct_values'],
                    'distinct_values_same': result1[col]['distinct_values'] == result2[col]['distinct_values']
                })

    # Convert comparison to DataFrame for serializability
    return pd.DataFrame(comparison)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
