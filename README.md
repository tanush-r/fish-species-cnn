# Deep Learning: Recognising Fish Species using CNN 

In this deep learning project, we are using Convolutional Neural Networks(using MobileNet on Kera), to recognise the species of the given fish. Using REST API we can provide the required image to our python server, and it will return an array of probabilites for the 9 supported fish types.

## Fish Types

* Black Sea Sprat

![black-sea-sprat](res/black-sea-sprat.png)

* Gilt Head Bream

![gilt-head-bream](res/gilt-head-bream.png)

* Hourse Mackerel

![hourse-mackerel](res/hourse-mackerel.png)

* Red Mullet

![red-mullet](res/red-mullet.png)

* Red Sea Bream

![red-sea-bream](res/red-sea-bream.png)

* Sea Bass

![sea-bass](res/sea-bass.png)

* Shrimp

![shrimp](res/shrimp.png)

* Striped Red Mullet

![striped-red-mullet](res/striped-red-mullet.png)

* Trout

![trout](res/trout.png)


## Getting Started

* Run [server.py]() to start the REST API
* Use Postman/cURL on command-line to pass the image (localhost/predict)
* The REST API will return a JSON

Output format:

```
{
    "Predictions" : [
        {"label":fish_name,"probability":prob},
        {"label":fish_name,"probability":prob},
        ...],
    "success" : true
    }
```

## Built With

* [VSCode](https://code.visualstudio.com/) - IDE used
* [Kaggle](https://www.kaggle.com/fahadmehfoooz/fish-analysis) - Dataset used

## Authors

* **Tanush R** - [tanush-r](https://github.com/tanush-r)

