def replacenan(t):
	return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from approximation_generation_file import *
from runner_utils import get_save_formatted_to_file
import joblib

def driver():
	x = np.arange(-100,100, 0.01, dtype=float)
	y = tf.math.abs(tf.cast(x, tf.float32))
	y = replacenan(y)
	poly = PolynomialFeatures(degree=29, include_bias=False)
	poly_features = poly.fit_transform(x.reshape(-1, 1))

	poly_reg_model = LinearRegression()
	poly_reg_model.fit(poly_features, y)
	y_predicted = poly_reg_model.predict(poly_features)

	mse = str(np.sqrt(mean_squared_error(y,y_predicted)))
	get_save_formatted_to_file("./supporting_files/tf.math.abs29_mse.txt", mse)

	joblib.dump(poly_reg_model, 'abs_aprox_model')
	joblib.dump(poly, 'polynomial_features_model')
	polynomia_features_model = joblib.load('polynomial_features_model')
	themodel = joblib.load('abs_aprox_model')
	X_val_prep = polynomia_features_model.transform(x.reshape(-1, 1))
	predictions = themodel.predict(X_val_prep)
	function_body_generation(poly_reg_model, rank=1, operation_name="tf.math.abs")

driver()