## Reproducibility package of: Addressing Conversion Faults in Machine Learning Models for Mobile and IoT Devices via Automatic Approximation
### This reproducibility package the following folders.
```
>./images-n-graphs
>./spreadsheets
>./sample-generated-source/extracted operation
>./sample-generated-source/generated approximation function
>./sample-generated-source/generated approximation model
>./sample-generated-source/regression generation
```

1. *[spreadsheets](spreadsheets)* -- contains two spreadsheets. 
 * [tensorflow-operations.xlsx](spreadsheets/tensorflow-operations.xlsx) -- contains the collected TensorFlow operations and if they are supported by the TensorFlow Lite converter.
 * [experimental-results.xlsx](spreadsheets/experimental-results.xlsx) -- contains all the data for the evaluation of Seamstress and its approximations.
2. *[sample-generated-source](sample-generated-source)* -- contains samples of automatically generated source code for the abs function
 * [extracted operation](sample-generated-source/extracted%20operation) -- contains the generated code snippet extracted from the TensorFlow repository
 * [regression generation](sample-generated-source/regression%20generation) - contains the code snippet that calculates a regression function for the ``abs'' operation.
 * [generated approximation function](sample-generated-source/generated%20approximation%20function) -- contains the generated approximation function by Seamstress
 * [generated approximation model](sample-generated-source/generated%20approximation%20model) -- contains two approximation models (abs_aprox). The *.pb is in the TensorFlow format. The *.tflite is the resulting conversion to TensorFlow Lite.
3. *[images-n-graphs](images-n-graphs)* -- contains high-resolution versions of the graphs in the manuscript
 * [avg-allocation-time.pdf](images-n-graphs/avg-allocation-time.pdf) -- Graph containing the average allocation time for all operations
 * [inference-time-mse-number-of-terms.pdf](images-n-graphs/inference-time-mse-number-of-terms.pdf) -- Graph containing mean square error, inference time, number of terms for all operations
 * [overview.pdf](images-n-graphs/overview.pdf) -- contains Seamstress overview image
 * [power-consumption.pdf](images-n-graphs/power-consumption.pdf) -- Graph containing power consumption information
 
 ### We will provide all the sources after rebuttal.
