'''
    Written By: Mehdi Touyserkani - Aug 2024.
    https://ir-bestpro.com.
    https://www.linkedin.com/in/bestpro-group/
    https://github.com/irbestpro/
    ir_bestpro@yahoo.com
    BESTPRO SOFTWARE ENGINEERING GROUP
'''

from pydantic import BaseModel , ConfigDict

class Params(BaseModel):
    model_config = ConfigDict(validate_assignment = True)
    EPOCH : int = 150
    LAG : int =  120
    LEARNING_RATE : float = 0.005
    HIDDEN_SIZE : int = 128
    NUM_OF_LAYERS :int = 1
    TRAINING_SIZE : float =  0.8
