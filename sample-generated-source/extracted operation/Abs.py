import tensorflow as tf
from tensorflow.python.framework import dtypes
class TestAbs(tf.Module):
	@tf.function
	def __call__(self, input):
		return tf.raw_ops.Abs(x=input)

def driver():

	a = tf.constant([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], shape=[2, 3])
	b = tf.constant([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], shape=[3, 2])
	c = tf.constant([0.5, 0.8, 0.251, 1.0], shape=[2, 2])

	model_abs = TestAbs()
	concrete_function = model_abs.__call__.get_concrete_function(c)
	graph_cf = concrete_function.graph.as_graph_def()
	write_graph("./gen_models/abs.pb", graph_cf)

	converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
	converter.experimental_new_converter=True
	converter.allow_custom_ops = True
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

	tflite_abs = converter.convert()

	write_model_to_file("./gen_models/abs.tflite", tflite_abs)
	interpreter_abs = tf.lite.Interpreter("./gen_models/abs.tflite")
	interpreter_abs.allocate_tensors()

def write_model_to_file(FILE_PATH, tfl_model):
	with open(FILE_PATH, 'wb') as f:
		f.write(tfl_model)
def write_graph(frozen_graph_filename, graphdef):
	tf.io.write_graph(graphdef, './', frozen_graph_filename)
driver()
