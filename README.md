# Simple Machine Learning as a Service (MLaaS)

Let's predict `Iris` species from Iris's metrics!!!

![overall](assets/overall.png)

## Iris's species

![iris-species](assets/iris-species.png)

## Iris's metrics

 - Sepal length
 - Sepal width
 - Petal length
 - Petal width
 - Species

note : 

 - Sepal(กลีบเลี้ยง)
 - Petal(กลีบดอก)

![iris-metrics](assets/iris-metrics.png)

## Installation

```bash
pip install -r requirements.txt
```

## Create Simple model to predict Iris

Load Iris data set from Sci-Kit learn datasets

Ref. [Jupyter notebook](create-model.ipynb)

```py
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris['data'], iris['target']
```

### Reshape data

```py
from sklearn.model_selection import train_test_split
import numpy as np

dataset = np.hstack((X, y.reshape(-1,1)))
np.random.shuffle(dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset[:,:4],
                                                   dataset[:,4],
                                                   test_size=0.2)
```

### Train model

In this example, I using `LogisticRegression` model :

```py
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Check accuracy

```py
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

### Save/Export the model

```py
import joblib

joblib.dump(model, 'iris.model')
```

## Simple Service with Flask

Example code :

```py
from flask import Flask, request
from flask_cors import CORS, cross_origin
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/iris', methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    inputs = np.array(req.split(','), dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'
    else:
        return 'Virginica'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## Iris's metrics for testing

| Sepal length | Sepal width | Petal length | Petal width |
|--------------|-------------|--------------|-------------|
|   5.1        |     3.5     |     1.4      |     0.2     |

It should predict to `Setosa`.

Example the request:

![request-predict](assets/ex-post-mlaas.png)

## License

MIT