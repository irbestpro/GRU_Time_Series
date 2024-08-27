from pydantic import BaseModel , ConfigDict

class Params(BaseModel):
    model_config = ConfigDict(validate_assignment = True)
    EPOCH : int = 101
    LAG : int =  5
    LEARNING_RATE : float = 0.005
    BATCH_SIZE : int = 5
    HIDDEN_SIZE : int = 128
    NUM_OF_LAYERS :int = 1
    TRAINING_SIZE : float =  0.8
    TEST_SIZE : float = 0.2