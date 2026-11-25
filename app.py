import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from features import AcademicFeatureEngineer

app = Flask(__name__)

bundle = joblib.load('student_models.joblib')
with open('model_artifacts.json', 'r') as f:
    artifacts = json.load(f)

reg_model = bundle['reg_model']
log_model = bundle['log_model']
knn_model = bundle['knn_model']
nn_preprocessor = bundle['nn_preprocessor']
nn_finder = bundle['nn_finder']
reference_students = bundle['reference_students']
kmeans_pipe = bundle['kmeans_pipe']
cluster_label_map = bundle['cluster_label_map']

@app.route('/')
def home():
    return render_template(
        'index.html',
        metrics=artifacts['metrics'],
        major_factors=artifacts['major_factors']
    )

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            'Attendance': float(data['attendance']),
            'Study_Hours': float(data['study_hours']),
            'Previous_Marks': float(data['previous_marks']),
            'Assignment_Marks': float(data['assignment_marks']),
            'Internal_Marks': float(data['internal_marks']),
            'Learning_Support': str(data['learning_support']).strip()
        }])

        predicted_marks = float(np.clip(reg_model.predict(input_df)[0], 0, 100))
        pass_pred = int(log_model.predict(input_df)[0])
        pass_prob = float(log_model.predict_proba(input_df)[0][1]) * 100
        fail_prob = 100.0 - pass_prob

        knn_pred = int(knn_model.predict(input_df)[0])

        cluster_id = int(kmeans_pipe.predict(input_df)[0])
        segment = cluster_label_map[cluster_id]

        if fail_prob >= 50 or predicted_marks < 45:
            risk_level = 'High'
            recommendation = 'Immediate academic counseling, attendance monitoring, and remedial tutoring recommended.'
        elif fail_prob >= 20 or predicted_marks < 62:
            risk_level = 'Medium'
            recommendation = 'Schedule weekly faculty check-ins and targeted practice on internal assessment topics.'
        else:
            risk_level = 'Low'
            recommendation = 'Student is on track. Encourage advanced projects and peer mentoring.'

        scaled_input = nn_preprocessor.transform(input_df)
        distances, indices = nn_finder.kneighbors(scaled_input)
        similar_peers = []
        for idx, dist in zip(indices[0], distances[0]):
            peer = reference_students.iloc[idx]
            similar_peers.append({
                'attendance': float(peer['Attendance']),
                'study_hours': float(peer['Study_Hours']),
                'previous_marks': float(peer['Previous_Marks']),
                'final_marks': float(peer['Final_Exam_Marks']),
                'status': str(peer['Status']),
                'similarity': round(max(0.0, 100.0 - float(dist) * 15.0), 1)
            })

        return jsonify({
            'predicted_marks': round(predicted_marks, 1),
            'pass_fail': 'Pass' if pass_pred == 1 else 'Fail',
            'knn_prediction': 'Pass' if knn_pred == 1 else 'Fail',
            'pass_probability': round(pass_prob, 1),
            'fail_probability': round(fail_prob, 1),
            'risk_level': risk_level,
            'segment': segment,
            'recommendation': recommendation,
            'similar_peers': similar_peers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
