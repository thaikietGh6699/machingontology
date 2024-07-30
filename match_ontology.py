import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, XSD  

# Read the CSV file containing calculated features
input_file = 'dataframe/features_101-302.csv'
features = pd.read_csv(input_file)

types = [1 if row == 'Class' else 0 for row in features['Type']]
features['Type_encode'] = types

X = features.loc[:, 'Ngram1_Entity':'Type_encode']

# Fill NaN values with 0
X = X.fillna(value=0)

# LogisticRegression or RandomForest or XGBoost
model_name = 'RandomForest'
print(f"Using model {model_name} to maching")
with open(f'models/{model_name}.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict with the pre-trained model
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

# Assign predictions to the 'Match' column in the features dataframe
features['Match'] = predictions
num_matched = features['Match'].sum()
total_rows = len(features)

print(f"Using threshold {threshold}:")
print(f"Value of matching: {num_matched}")

output_file = "dataframe/matching_101-301.csv"
features.to_csv(output_file, index=False)

matched_rows = features[features['Match'] == 1]

num_matched = len(matched_rows)
print(f"Number of rows with 'Match' value 1: {num_matched}")


# Swing rdf alignment
g = Graph()
align = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#")

alignment = URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#Alignment")
g.add((alignment, RDF.type, align.Alignment))
g.add((alignment, align.xml, Literal("yes")))
g.add((alignment, align.level, Literal("0", datatype=XSD.integer)))
g.add((alignment, align.type, Literal("11", datatype=XSD.integer)))
g.add((alignment, align.onto1, URIRef("http://oaei.ontologymatching.org/tests/101/onto.rdf")))
g.add((alignment, align.onto2, URIRef("http://oaei.ontologymatching.org/tests/302/onto.rdf")))
g.add((alignment, align.uri1, URIRef("http://oaei.ontologymatching.org/tests/101/onto.rdf")))
g.add((alignment, align.uri2, URIRef("http://ebiquity.umbc.edu/v2.1/ontology/publication.owl")))

for index, row in matched_rows.iterrows():
    cell = URIRef(f"cell_{index}")
    entity1 = URIRef(row['Ontology1'] + "#" + row['Entity1'])
    entity2 = URIRef(row['Ontology2'] + "#" + row['Entity2'])
    measure = Literal(row['Cosine_similarity_Entity'], datatype=XSD.float)
    relation = Literal("=", datatype=XSD.string)

    g.add((alignment, align.map, cell))
    g.add((cell, align.entity1, entity1))
    g.add((cell, align.entity2, entity2))
    g.add((cell, align.measure, measure))
    g.add((cell, align.relation, relation))
    
algm = "alignment_101-302.rdf"
g.serialize(destination=algm, format='xml')
print(f"Alignment RDF file saved to {algm}")

# Calculate Jaccard Index
from rdflib import Graph, RDF, OWL
s = Graph()
t = Graph()
s.parse("dataset1/ontologies/101.rdf", format="xml")
t.parse("dataset1/ontologies/302.rdf", format="xml")

classes_s = len(set(s.subjects(predicate=RDF.type, object=OWL.Class)))
properties_s = len(set(s.subjects(predicate=RDF.type, object=OWL.Property)))
print(f"Number of classes in ontology 101: {classes_s}")
print(f"Number of properties in ontology 101: {properties_s}")

classes_t = len(set(t.subjects(predicate=RDF.type, object=OWL.Class)))
properties_t = len(set(t.subjects(predicate=RDF.type, object=OWL.ObjectProperty)))
print(f"Number of classes in ontology 302: {classes_t}")
print(f"Number of properties in ontology 302: {properties_t}")

O_S = classes_s + properties_s
O_T = classes_t + properties_t
print("Ontology source:", O_S)
print("Ontology target:", O_T)

# Ensure no division by zero and valid Jaccard Index calculation
M = features['Match'].sum()
if O_S + O_T - M > 0:
    jaccard_index = M / (O_S + O_T - M)
else:
    jaccard_index = 0
    print("The two ontologies don't match")

jaccard_index_percent = jaccard_index * 100

print(f"corresponding ratio between 2 ontology: {jaccard_index_percent:.2f}%")