import os
from flask import Flask, render_template, request
import pandas as pd
import yaml
import pickle
import xgboost as xgb
from rdflib import Graph, RDF, OWL
import re


app = Flask(__name__)

# Đường dẫn lưu trữ tạm thời các file tải lên
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train_view():
    if request.method == 'POST':
        # Lưu các file tải lên
        config_file = request.files['config']
        train_csv_file = request.files['train_csv']
        test_csv_file = request.files['test_csv']

        config_path = os.path.join(app.config['UPLOAD_FOLDER'], 'config.yml')
        train_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
        test_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test.csv')

        config_file.save(config_path)
        train_csv_file.save(train_csv_path)
        test_csv_file.save(test_csv_path)

        # Đọc file cấu hình
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        SELECTED_DATASET = config['DATASET']
        SELECTED_MODEL = config['MODEL']

        train_features = pd.read_csv(train_csv_path)
        test_features = pd.read_csv(test_csv_path)

        # Tạo feature "Type" cho tập huấn luyện và kiểm tra
        train_features['Type_encode'] = train_features['Type'].apply(lambda x: 1 if x == 'Class' else 0)
        test_features['Type_encode'] = test_features['Type'].apply(lambda x: 1 if x == 'Class' else 0)

        X_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
        y_train = train_features['Match']

        X_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
        y_test = test_features['Match']

        # Điền giá trị NaN bằng 0
        X_train = X_train.fillna(value=0)
        X_test = X_test.fillna(value=0)

        trained_models = {}

        # Huấn luyện mô hình
        if SELECTED_MODEL != 'XGBoost':
            if SELECTED_MODEL == 'LogisticRegression':
                print("Training logistic regression...")
                from sklearn.linear_model import LogisticRegression

                if SELECTED_DATASET == 'dataset1':
                    model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)
                elif SELECTED_DATASET == 'dataset2':
                    model = LogisticRegression(penalty='l2', C=7.742637, class_weight=None)
                
            elif SELECTED_MODEL == 'RandomForest':
                print("Training random forest classifier...")
                from sklearn.ensemble import RandomForestClassifier

                if SELECTED_DATASET == 'dataset1':
                    model = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=3, random_state=42)
                elif SELECTED_DATASET == 'dataset2':
                    model = RandomForestClassifier(n_estimators=100, max_features=None, max_depth=2)

            model.fit(X_train, y_train)
            trained_models[SELECTED_MODEL] = model
            print("Predicting for testing dataset...")
            y_prob = model.predict_proba(X_test)

        elif SELECTED_MODEL == 'XGBoost':
            import xgboost as xgb

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            param = {'silent': 0, 'objective': 'binary:logistic',
                    'min_child_weight': 10, 'gamma': 2.0, 'subsample': 0.8,
                    'colsample_bytree': 0.8, 'max_depth': 5, 'nthread': 4,
                    'eval_metric': 'error'}

            evallist = [(dtest, 'eval'), (dtrain, 'train')]

            num_round = 10
            bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)
            trained_models[SELECTED_MODEL] = bst

            y_prob = bst.predict(dtest)
        
        # Lưu các mô hình đã huấn luyện
        for model_name, trained_model in trained_models.items():
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f'{model_name}.pkl'), 'wb') as file:
                pickle.dump(trained_model, file)

        TEST_ALIGNMENTS = config[SELECTED_DATASET]['TEST_ALIGNMENTS']

        # Chọn ngưỡng tốt nhất
        results = []
        for alignment in TEST_ALIGNMENTS:
            ont1 = alignment.split('-')[0]
            ont2 = alignment.split('-')[1].replace('.rdf', '')
            best_ts = 0
            best_fmeasure = 0

            for ts in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                preds = []
                if SELECTED_MODEL != 'XGBoost':
                    preds = [1 if x[1] >= ts else 0 for x in y_prob]
                else:
                    preds = [1 if x >= ts else 0 for x in y_prob]

                test_features['Predict'] = preds

                if SELECTED_DATASET == 'dataset1':
                    onto_format = 'rdf'
                elif SELECTED_DATASET == 'dataset2':
                    onto_format = 'owl'

                pred_mappings = test_features[
                    (test_features['Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                    (test_features['Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                    (test_features['Predict'] == 1)]

                true_mappings = test_features[
                    (test_features['Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                    (test_features['Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                    (test_features['Match'] == 1)]

                correct_mappings = test_features[
                    (test_features['Ontology1'] == f"{SELECTED_DATASET}/ontologies/{ont1}.{onto_format}") &
                    (test_features['Ontology2'] == f"{SELECTED_DATASET}/ontologies/{ont2}.{onto_format}") &
                    (test_features['Match'] == 1) & (test_features['Predict'] == 1)]

                true_num = len(true_mappings)
                predict_num = len(pred_mappings)
                correct_num = len(correct_mappings)

                if predict_num != 0 and true_num != 0 and correct_num != 0:
                    precision = correct_num / predict_num
                    recall = correct_num / true_num
                    fmeasure = 2 * precision * recall / (precision + recall)
                else:
                    fmeasure = 0

                if fmeasure > best_fmeasure:
                    best_fmeasure = fmeasure
                    best_ts = ts
                    best_preds = preds
                    
            selected_model = SELECTED_MODEL
            
            results.append(
                f"Best fmeasure for {alignment} is {best_fmeasure} with threshold {best_ts}"
            )

        return render_template('train.html', results=results , selected_model=selected_model)

    return render_template('train.html')

@app.route('/match', methods=['GET', 'POST'])
def match_view():
    model_name = None
    M = None
    num_matched = None
    O_S = None
    O_T = None
    jaccard_index = None
    jaccard_index_percent = None
    
    if request.method == 'POST':
        input_file = request.files['input_file']
        model_name = request.form['model']
        
        # Đọc tệp CSV đầu vào và tạo các tính năng cần thiết
        features = pd.read_csv(input_file)
        types = [1 if row == 'Class' else 0 for row in features['Type']]
        features['Type_encode'] = types
        X = features.loc[:, 'Ngram1_Entity':'Type_encode']
        X = X.fillna(value=0)

        # Load mô hình đã được huấn luyện
        with open(f'models/{model_name}.pkl', 'rb') as file:
            model = pickle.load(file)

        # Dự đoán với mô hình đã được huấn luyện
        if model_name != 'XGBoost':
            y_prob = model.predict_proba(X)
        else:
            dmatrix = xgb.DMatrix(X)
            y_prob = model.predict(dmatrix)

        threshold = 0.5
        if model_name != 'XGBoost':
            predictions = (y_prob[:, 1] > threshold).astype(int) 
        else:
            predictions = (y_prob > threshold).astype(int)

        # Gán dự đoán vào cột 'Match' trong dataframe features
        features['Match'] = predictions

        # Trích xuất số ontology từ tên tệp đầu vào
        numbers = re.findall(r'\d+', input_file.filename)

        if len(numbers) == 2:
            first_number = numbers[0]
            second_number = numbers[1]

            # Xây dựng các chuỗi ontology tương ứng
            ontology_1 = f"dataset1/ontologies/{first_number}.rdf"
            ontology_2 = f"dataset1/ontologies/{second_number}.rdf"

            # Parse các ontology
            s = Graph()
            t = Graph()
            s.parse(ontology_1, format="xml")
            t.parse(ontology_2, format="xml")

            # Tính toán chỉ số Jaccard
            classes_s = len(set(s.subjects(predicate=RDF.type, object=OWL.Class)))
            properties_s = len(set(s.subjects(predicate=RDF.type, object=OWL.ObjectProperty)))
            classes_t = len(set(t.subjects(predicate=RDF.type, object=OWL.Class)))
            properties_t = len(set(t.subjects(predicate=RDF.type, object=OWL.ObjectProperty)))
            O_S = classes_s + properties_s
            O_T = classes_t + properties_t
            M = features['Match'].sum()
            if O_S + O_T - M > 0:
                jaccard_index = M / (O_S + O_T - M)
            else:
                jaccard_index = 0

            jaccard_index_percent = jaccard_index * 100

            # Lưu dataframe đã được xử lý và in kết quả
            output_file = f"dataframe/matching_{input_file.filename}"
            features.to_csv(output_file, index=False)
            matched_rows = features[features['Match'] == 1]
            num_matched = len(matched_rows)
            print(f"Value of matching: {M}")
            print(f"Jaccard Index: {jaccard_index_percent:.2f}%")

    return render_template('match.html', model_name=model_name, match_value=M, num_matched=num_matched, O_S=O_S, O_T=O_T, jaccard_index=jaccard_index_percent)

if __name__ == '__main__':
    app.run(debug=True)
