from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import openai
import os
import uuid
import shutil
import glob

my_secret = os.environ['keygpt']

openai.api_key = my_secret

def get_special_characters(values):
    special_chars = set()
    for value in values:
        value_str = str(value)  # Convert value to string
        for char in value_str:
            if not char.isalnum():  # Identify non-alphanumeric characters
                special_chars.add(char)
    return special_chars
  
app = Flask(__name__)
app.secret_key = my_secret  # change this to a real secret key

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  
    folder_path = 'temp_files/'
    files = glob.glob(folder_path + '*')
    # Supprimer chaque fichier
    for f in files:
      os.remove(f)
  
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']

        df1 = pd.read_csv(file1.stream, sep=';')
        df2 = pd.read_csv(file2.stream, sep=';')

        result1 = analyse_df(df1)
        result2 = analyse_df(df2)

        comparison = compare_results(result1, result2, df1, df2)
        filename = str(uuid.uuid4()) + ".json"  # Génère un nom de fichier unique
        filepath = os.path.join("temp_files", filename)  # Sauvegarde dans un répertoire temp_files
        comparison.to_json(filepath)
        session['comparison_file'] = filepath
  
        session['filename1'] = file1.filename
        session['filename2'] = file2.filename

        

        return redirect(url_for('results'))  # Redirect to the results route

    return render_template('upload.html')


@app.route('/results', methods=['GET'])
def results():
    filepath = session.get('comparison_file', None)
    if filepath is None or not os.path.exists(filepath):
        return redirect(url_for('upload_file'))  # Redirecting back to the upload page as an example

    comparison = pd.read_json(filepath)

    # Suppression du fichier temporaire
    os.remove(filepath)

    filename1 = session.get('filename1', "File 1")
    filename2 = session.get('filename2', "File 2")
    return render_template('results.html', comparison=comparison, filename1=filename1, filename2=filename2)




def analyse_df(df):
    result = {}
    for col in df.columns:
        # Detect column type
        detected_type = df[col].dtypes

        if pd.api.types.is_bool_dtype(df[col]):
            # For boolean columns
            value_type = 'boolean'
            #numeric_sum = None
        elif np.issubdtype(detected_type, np.number):
            # For numeric columns
            value_type = 'numeric'
            numeric_sum = df[col].sum()  # calculate the sum of numeric values
        elif detected_type == 'object':
            # For string columns
            value_type = 'object'
            temp_col = df[col].replace(',', '.', regex=True)
            numeric_values = pd.to_numeric(temp_col, errors='coerce')
  # try to convert to numeric
            numeric_sum = numeric_values.sum()  # calculate the sum of numeric values
        elif np.issubdtype(detected_type, np.datetime64):
            # For date columns
            value_type = 'datetime'
            numeric_sum = None
        else:
            value_type = 'unknown'
            numeric_sum = None

        #if not np.issubdtype(df[col].dtype, np.number):
         #  distinct_colvalues = df[col].unique().tolist()
        #else:
         #  distinct_colvalues = None
        distinct_colvalues = ['vide' if (isinstance(x, (float, np.float64)) and np.isnan(x)) else x for x in set(df[col].unique())]
        result[col] = {
            'type': value_type,
            'detected_type': str(detected_type),
            'total_values': df[col].count(),
            'null_values': df[col].isnull().sum(),
            'numeric_sum': numeric_sum,  # add the sum of numeric values
            'distinct_colvalues' :  distinct_colvalues,
             
                
        }
        #print(distinct_colvalues)

        # If column is of type 'object', 'float64', or 'boolean', calculate distinct values
        if (detected_type) in ['object', 'float64', 'bool','boolean','int64']:
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
                'null_values_same': result1[col]['null_values'] == result2[col]['null_values'],
                'numeric_sum_file1': result1[col]['numeric_sum'] if 'numeric_sum' in result1[col] else 'N/A',
                'numeric_sum_file2': result2[col]['numeric_sum'] if 'numeric_sum' in result2[col] else 'N/A',
                'numeric_sum_same': abs(result1[col]['numeric_sum'] - result2[col]['numeric_sum']) < 1e-9 
              #if 'numeric_sum' in result1[col] and 'numeric_sum' in result2[col] else 'N/A',

              
              }
            
            if 'distinct_values' in result1[col] and 'distinct_values' in result2[col]:
                
              comparison[col].update({
                    'distinct_values_file1': result1[col]['distinct_values'],
                    'distinct_values_file2': result2[col]['distinct_values'],
                    'distinct_values_same': result1[col]['distinct_values'] == result2[col]['distinct_values']
                })

           # if result1[col]['detected_type'] != result2[col]['detected_type']:
            #    detected_type1 = result1[col]['detected_type']
            #    detected_type2 = result2[col]['detected_type']
             #   sample1 = df1[col].dropna().sample(min(5, len(df1[col].dropna()))).tolist()  # sample from file1
            #    sample2 = df2[col].dropna().sample(min(5, len(df2[col].dropna()))).tolist()  # sample from file2
            #    sample1_str = [str(val) if isinstance(val, bool) else val for val in sample1]
            #    sample2_str = [str(val) if isinstance(val, bool) else val for val in sample2]
              
                #prompt = f"In 200 words maximum, why might pandas detect the values {sample1_str} with the detected type '{detected_type1}' and the values {sample2_str} with the detected type '{detected_type2}' by illustrating with the values?"
                #response = openai.Completion.create(
                   # engine="text-davinci-002",
                   # prompt=prompt,
                   # max_tokens=300
               # )


               # comparison[col].update({
                #    'reason_for_different_types': response.choices[0].text.strip()
               # })

              
            if result1[col]['detected_type'] != 'number' and result2[col]['detected_type'] != 'number':
                distinct_colvalues1 = result1[col]['distinct_colvalues']
                distinct_colvalues2 = result2[col]['distinct_colvalues']
                unique_values1 = set(distinct_colvalues1) - set(distinct_colvalues2)
                unique_values2 = set(distinct_colvalues2) - set(distinct_colvalues1)

                # Get unique special characters
                special_chars_col1 = get_special_characters(unique_values1)
                special_chars_col2 = get_special_characters(unique_values2)

                distinct_colvalues1_without_nulls = set(value for value in distinct_colvalues1 if value is not float('nan'))
                print(distinct_colvalues1_without_nulls)
                distinct_colvalues2_without_nulls = set(value for value in distinct_colvalues2 if value is not float('nan'))
                

                comparison[col].update({
                    'Distinct_colvalues1': distinct_colvalues1,
                    'Distinct_colvalues2': distinct_colvalues2,
                    'Distinct_colvalues_same': distinct_colvalues1_without_nulls == distinct_colvalues2_without_nulls,
                    'Distinct_colvalues1_only': list(unique_values1),
                    'Distinct_colvalues2_only': list(unique_values2),
                    'UniqueSpecialChars1': list(special_chars_col1),
                    'UniqueSpecialChars2': list(special_chars_col2),
                })
              
            validation = (
                comparison[col]['detected_type_same'] and
                comparison[col]['total_values_same'] and
                comparison[col]['null_values_same'] and
                comparison[col]['distinct_values_same'] and
                comparison[col]['numeric_sum_same'] and
                comparison[col]['Distinct_colvalues_same']
            )
            
            # Add validation to this column's data
            comparison[col]['Validation'] = validation
      
    # Convert comparison to DataFrame for serializability
    for col, col_data in comparison.items():
        for key, value in col_data.items():
            if value is None:
                comparison[col][key] = "None"
              
            #print(pd.DataFrame(comparison))
    return pd.DataFrame(comparison)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)