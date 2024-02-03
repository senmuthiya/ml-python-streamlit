import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import os

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Streamlit app
st.title("Iris Classification")

# Set parameters
st.subheader("Set Parameters")
test_size = st.slider("Test Set Size (%)", 0, 100, 20, 5)
random_state = st.slider("Random State", 0, 100, 42, 1)

# Handle the case when test_size is set to 0
if test_size == 0:
    test_size = 0.01  # Set a small positive value

# Train-test split
if test_size == 100:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=random_state)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Display model performance directly in the main page
show_accuracy = st.checkbox("Show Model Accuracy")
if show_accuracy:
    accuracy = accuracy_score(y_test, y_pred)
    # st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix directly in the main page
show_confusion_matrix = st.checkbox("Show Confusion Matrix")
if show_confusion_matrix:
    # st.subheader("Confusion Matrix")
    confusion_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    st.write(confusion_mat)

# Allow the user to make predictions on new data directly in the main page
st.subheader("Make Predictions")
sepal_length = st.slider("Sepal Length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()), float(X['sepal length (cm)'].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()), float(X['sepal width (cm)'].mean()))
petal_length = st.slider("Petal Length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()), float(X['petal length (cm)'].mean()))
petal_width = st.slider("Petal Width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()), float(X['petal width (cm)'].mean()))

# Create a DataFrame with feature names for prediction
user_input = pd.DataFrame(
    {'sepal length (cm)': [sepal_length],
     'sepal width (cm)': [sepal_width],
     'petal length (cm)': [petal_length],
     'petal width (cm)': [petal_width]}
)

# Make a prediction
prediction = model.predict(user_input)[0]

# Display the predicted class above the image display
predicted_class_text = f"Predicted: {iris.target_names[prediction]}"
st.subheader(predicted_class_text)

# Set the working directory to the location of your script
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Display the respective image based on the predicted class
image_path = f"images/{iris.target_names[prediction].lower()}.jpg"
try:
    image = Image.open(image_path)
    st.image(image, use_column_width=False, width=200, clamp=True)
except FileNotFoundError:
    st.warning(f"Image not found: {image_path}")

# Display the dataset based on user selection directly in the main page
show_full_data = st.checkbox("Show Full Dataset")
if show_full_data:
    st.dataframe(X)
else:
    st.dataframe(X.head())
