## A simple AI project template for classification.

To run:

First, you need to run 
```
conda create
```
Then you need to prepare the data sample by running:
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

A report also will help people to understand your methodology and results. 
- You can start by giving a brief overview of the problem and what people have done before. We can call this part **Introduction**. 
- Then, you can write about the data set. How you prepared. You need to describe it and put the relevant reference if you have downloaded the data.
- Next, you better write about what you have done and why you have done it. The way you prepared the data, preprocessing steps, the training procedure, etc. should be mentioned here. This section is called **Methodology**. 
- The most interesting part is called **Results**. In this section, you should write about your results, visualize them, and interpret them. You need to answer this question: "what did one learn from these results?". You also may need to compare your results with other related efforts. You also should mention if there is any advantage in your results compared to others. 
- You also need to tell a brief story about why the problem is important and what one can learn about your experiment as a **conclusion**. This section should be short, and it needs to include the whole story in a highly informative way.  
- Last but definitely not least, you need to write an **abstract**. This section is where you can give most of the information and the first impression (after the title and figures!). It better ends with a quantitative description of why your results should be published. 



