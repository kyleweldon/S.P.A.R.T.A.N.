#############################################
# File created by Kyle Weldon on 04/05/2024 #
#############################################

import AIModel
import MeaningsAtIndecies
import numpy as np

################### Questions/Talking points for next meeting ##########################
'''
- Why do some people have a 0 as their decision instead of a 1 or 2? (3a 96ish)
'''
########################################################################################
################# Needs Doing list for entire project ##############
# TODO: Combine functions and loops so there is least amount of repeated code as possible in all files
# TODO: Comment all the code in all the files
# TODO: Add variables into the main program to control more about the model as perameters
# TODO: Finish Multi_inputModel class and main code implementation
#############################################################

def main(**CONTROLS):
    output_index_at_decision = MeaningsAtIndecies.get_output_meanings()

    for index, decision in output_index_at_decision.items():
        if decision in CONTROLS['decisions_to_fit']:
            print('=' * 120)

            print(f"Decision {decision}")

            # Calling correct model functiton
            if CONTROLS['MODEL'] == 0:
                model_obj = AIModel.SingleInputModel(index)
            elif CONTROLS['MODEL'] == 1:
                model_obj = AIModel.Multi_inputModel(index)
            elif CONTROLS['MODEL'] == 2:
                pass

            CallModelMethods(model_obj, CONTROLS['verbose'])

            # Releasing the object references to make the memory eligible
            # for the python garbage memory manager
            del model_obj
            model_obj = None

            print('=' * 120)

def CallModelMethods(model_obj, verbose):

    model_obj.compile_and_train_model()
    if verbose >= 1:
        model_obj.explain_model()
    if verbose >= 2:
        model_obj.evaluate_model()
    if verbose >= 3:
        model_obj.new_predictions()


####################
# Program Controls #
#####################
if __name__ == '__main__':
    '''
    MODEL 0 -> Sequential Model
    MODEL 1 -> Multi Input Model
    MODEL 2 -> Recurent Model (Not built yet)
    -----------------------------------------
    verbose 0 -> Put action here
    verbose 1 -> Put action here
    verbose 2 -> Put action here
    verbose 3 -> Put action here
    '''
    # Rename verbose variable to something that makes sense
    main(MODEL=1, decisions_to_fit=['1a'], verbose=0)
