# GRU_Time_Series
Stock Data Analysis by GRU model. <br/>

This model has two main steps: <br/>

1- normalizing the input values to make the model's performance better. <br/>
2- Using GRU units with a fully connected layer to predict the k-next data points in series.<br/>

To start the program, please run: <br/>
Python -m uvicorn Start:app --reload (macOS: python3 -m uvicorn Start:app --reload )<br/><br/>

* - Check the model's hyperparameters in ENV/parameters.py file and change them if needed.