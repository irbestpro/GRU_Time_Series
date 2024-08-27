import datetime
from fastapi import FastAPI , Request , status , Depends , File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse , StreamingResponse , JSONResponse
from fastapi.exceptions import HTTPException
from typing import Annotated
from functools import lru_cache
import json

from ENV.Params import Params
from Model.GRU_Model import Time_Series_Model

#________________Main FastAPI Object Here______________________

app = FastAPI()
Templates = Jinja2Templates(directory='./Templates/') # fast api templating

#________________________Root Path______________________________

@app.get('/' , response_class = HTMLResponse)
def Starter(request : Request):
    return Templates.TemplateResponse(request = request , name = "Index.html")

#______________Upload Dataset and Prediction____________________

async def Write_CSV_File(file : bytes): # co-routine for writing the csv file on server
    file_name = str(datetime.datetime.now()).replace(' ' , '-').replace(':' , '-') + '.csv'
    with open(f"./TempFiles/{file_name}" , 'wb') as File:
        File.write(file)
    return file_name

async def Upload_Dataset_File(dataset : Annotated [bytes | None , File()]): # upload csv file on server
    if not dataset:
        return None
    if len(dataset) > 400000 : # approximatyly 300KB
        return None
    file_name = await Write_CSV_File(dataset)
    return file_name

@lru_cache
def Return_Params(): # using pydantic Basemodel as model parameters
    return Params()

@app.post('/Time_Series_Prediction/')
async def Create_GRU_Based_Model(ack : Annotated[str|None , Depends(Upload_Dataset_File)] , parameters : Annotated[Params , Depends(Return_Params)]):
    if not ack: # async call to upload_dataset_file dependency
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail= "The maximum file size is 300KB in csv format!")
    
    model =  Time_Series_Model(DatasetName = ack,  # create object instance of time-series model
                                epoch = parameters.EPOCH ,
                                lag = parameters.LAG , 
                                train_size = parameters.TRAINING_SIZE , 
                                test_size = parameters.TEST_SIZE , 
                                learning_rate = parameters.LEARNING_RATE , 
                                num_layers = parameters.NUM_OF_LAYERS , 
                                hidden_size = parameters.HIDDEN_SIZE)

    #return StreamingResponse(model.Predict() , media_type='application/json')
    return StreamingResponse(model.Predict() , media_type='text/event-stream')

    
