import tensorflow as tf
from dataset_15mb import get_test_tensor
from tensorflow.python.framework import dtypes
class TestAproxAbs(tf.Module):
	@tf.function
	def __call__(self, x):
		return abs_poly_regression(x)

def abs_poly_regression(x):
	c1 = tf.constant(1, dtype=tf.float32)
	c2 = tf.constant(2, dtype=tf.float32)
	c3 = tf.constant(3, dtype=tf.float32)
	c4 = tf.constant(4, dtype=tf.float32)
	c5 = tf.constant(5, dtype=tf.float32)
	c6 = tf.constant(6, dtype=tf.float32)
	c7 = tf.constant(7, dtype=tf.float32)
	c8 = tf.constant(8, dtype=tf.float32)


	theta1 = tf.constant(2.4150314709316154e-05, dtype=tf.float32)
	theta2 = tf.constant(0.029608165165100654, dtype=tf.float32)
	theta3 = tf.constant(-2.987120307792652e-08, dtype=tf.float32)
	theta4 = tf.constant(-6.415111122939567e-06, dtype=tf.float32)
	theta5 = tf.constant(7.76517148955299e-12, dtype=tf.float32)
	theta6 = tf.constant(7.69812100570107e-10, dtype=tf.float32)
	theta7 = tf.constant(-5.551115123125783e-16, dtype=tf.float32)
	theta8 = tf.constant(-3.338431961430466e-14, dtype=tf.float32)

	pred0 = tf.constant(6.728869566817799, dtype=tf.float32)
	pred1 = tf.multiply(tf.pow(x, c1), theta1)
	pred2 = tf.multiply(tf.pow(x, c2), theta2)
	pred3 = tf.multiply(tf.pow(x, c3), theta3)
	pred4 = tf.multiply(tf.pow(x, c4), theta4)
	pred5 = tf.multiply(tf.pow(x, c5), theta5)
	pred6 = tf.multiply(tf.pow(x, c6), theta6)
	pred7 = tf.multiply(tf.pow(x, c7), theta7)
	pred8 = tf.multiply(tf.pow(x, c8), theta8)

	add1 = tf.add(pred0, pred1)
	add2 = tf.add(add1, pred2)
	add3 = tf.add(add2, pred3)
	add4 = tf.add(add3, pred4)
	add5 = tf.add(add4, pred5)
	add6 = tf.add(add5, pred6)
	add7 = tf.add(add6, pred7)
	add8 = tf.add(add7, pred8)

	return add8

def driver():

	a = tf.constant([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], shape=[2, 3])
	b = tf.constant([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], shape=[3, 2])
	c = get_test_tensor()

	model_abs_poly_regression = TestAproxAbs()
	concrete_function = model_abs_poly_regression.__call__.get_concrete_function(c)
	graph_cf = concrete_function.graph.as_graph_def()
	write_graph("abs_aprox.pb", graph_cf)

	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
	converter.experimental_new_converter=True
	converter.allow_custom_ops = True
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

	tflite_abs_aprox = converter.convert()

	write_model_to_file("./test_files/abs_aprox.tflite", tflite_abs_aprox)
	interpreter_abs_aprox = tf.lite.Interpreter("./test_files/abs_aprox.tflite")
	interpreter_abs_aprox.allocate_tensors()

def write_model_to_file(FILE_PATH, tfl_model):
	with open(FILE_PATH, 'wb') as f:
		f.write(tfl_model)
def write_graph(frozen_graph_filename, graphdef):
	tf.io.write_graph(graphdef, './test_files/', frozen_graph_filename)
driver()
