import joblib

loaded_object = joblib.load('models/Sales_regress.joblib')

# Now you can interact with the loaded object
print(loaded_object)
# For example, if it's a scikit-learn model:
# predictions = loaded_object.predict(new_data)