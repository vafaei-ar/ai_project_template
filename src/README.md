## MLflow with Train Script

To run:

```
mlflow run .
```

This will run `1-train.py` with the default parameters.

In order to run the file with custom parameters, run the command

```
mlflow run . \
-P experiment='TEST PROJECT' \
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


