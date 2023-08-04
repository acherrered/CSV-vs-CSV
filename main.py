from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import openai
import os

my_secret = os.environ['keygpt']

openai.api_key = my_secret

app = Flask(__name__)
app.secret_key = my_secret  # change this to a real secret key

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        df1 = pd.read_csv(file1.stream, sep=';')
        df2 = pd.read_csv(file2.stream, sep=';')

        result1 = analyse_df(df1)
        result2 = analyse_df(df2)

        comparison = compare_results(result1, result2, df1, df2)  # Pass df1 and df2 to compare_results

        # Save the results and filenames in session
        session['comparison'] = comparison.to_dict()
        session['filename1'] = file1.filename
        session['filename2'] = file2.filename

        return redirect(url_for('results'))  # Redirect to the results route

    return render_template('upload.html')


@app.route('/results', methods=['GET'])
def results():
    # Retrieve the results and filenames from session
    comparison = session.get('comparison', None)
    filename1 = session.get('filename1', "File 1")
    filename2 = session.get('filename2', "File 2")
    return render_template('results.html', comparison=comparison, filename1=filename1, filename2=filename2)


def analyse_df(df):
    result = {}
    for col in df.columns:
        # Detect column type
        detected_type = str(df[col].dtypes)

        if np.issubdtype(detected_type, np.number):
            # For numeric columns
            value_type = 'numeric'
        elif np.issubdtype(detected_type, np.bool_):
            # For boolean columns
            value_type = 'boolean'
        elif detected_type == 'object':
            # For string columns
            value_type = 'string'
        elif np.issubdtype(detected_type, np.datetime64):
            # For date columns
            value_type = 'datetime'

        result[col] = {
            'type': value_type,
            'detected_type': detected_type,
            'total_values': df[col].count(),
            'null_values': df[col].isnull().sum()
        }

        # If column is of type 'object', 'float64', or 'bool', calculate distinct values
        if detected_type in ['object', 'float64', 'bool']:
            result[col]['distinct_values'] = len(df[col].unique())

    # Convert result to DataFrame for serializability
    return pd.DataFrame(result)


def compare_results(result1, result2, df1, df2):
    comparison = {}
    for col in result1:
        if col in result2:  # check if column exists in both dataframes
            value1 = df1[col].mode().iloc[0] if not df1[col].isnull().all() else 'N/A'  # calculate mode for file1
            value2 = df2[col].mode().iloc[0] if not df2[col].isnull().all() else 'N/A'  # calculate mode for file2
            comparison[col] = {
                'type': result1[col]['type'],  # add column type
                'detected_type_file1': result1[col]['detected_type'],
                'detected_type_file2': result2[col]['detected_type'],
                'detected_type_same': result1[col]['detected_type'] == result2[col]['detected_type'],
                'most_common_value_file1': value1,
                'most_common_value_file2': value2,
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

            if result1[col]['detected_type'] != result2[col]['detected_type']:
                detected_type1 = result1[col]['detected_type']
                detected_type2 = result2[col]['detected_type']
                sample1 = df1[col].dropna().sample(min(5, len(df1[col].dropna()))).tolist()  # sample from file1print
                print(sample1)
                sample2 = df2[col].dropna().sample(min(5, len(df2[col].dropna()))).tolist()  # sample from file2
                print(sample2)
                sample1_str = [str(val) if isinstance(val, bool) else val for val in sample1]
                sample2_str = [str(val) if isinstance(val, bool) else val for val in sample2]
                print(sample1_str)
                print(sample2_str)
                prompt = f"in 200 words maximum Why might pandas detect the values {sample1_str} with the detected type '{detected_type1}' and the values {sample2_str} with the detected type '{detected_type2}' by illustrating with the values?"
                print(prompt)  # This will print the prompt
                response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=300
)

                comparison[col].update({
                    'reason_for_different_types': response.choices[0].text.strip()
                })

    # Convert comparison to DataFrame for serializability
    return pd.DataFrame(comparison)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
