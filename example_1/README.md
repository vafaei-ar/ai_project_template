## A simple AI project template for classification.

To run:

First, you need to run 
```
conda env create -f environment.yml
```
to make the environment and 
```
conda activate aitest
```
to get into it. Then you need to prepare the data sample by running:
```
python 0-preparation.py
```
It downloads the data into _data_ directory.
Then you should run:

```
mlflow run .
```

This will run `1-train.py` with the default parameters to train a model.

In order to run the file with custom parameters, you can run it like:

```
mlflow run . \
-P run_name='test' \
-P batch_size=64 \
-P epochs=20 \
-P aug_rot=45 \
-P aug_w=0.05 \
-P aug_h=0.05 \
-P aug_zoom=0.05 \
-P model_path='../models' \
--no-conda
```

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

After all and to predict with the trained model, you can run the _User Interface_ by:
```
streamlit run 2-predict.py
```



