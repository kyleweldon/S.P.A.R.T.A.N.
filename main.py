#############################################
# File created by Kyle Weldon on 04/05/2024 #
#############################################

import AIModel
import MeaningsAtIndecies
import numpy as np

################### Questions/Talking points for next meeting ##########################
'''
- Why do some people have a 0 as their decision instead of a 1 or 2? 
'''
########################################################################################
################# Needs Doing list for entire project ##############
# TODO: Change the methods_to_call perameter into a list and call all methods in list
# TODO: Fix the shap explainer so it works with multi input data
#############################################################

def main(**CONTROLS):
    output_index_at_decision = MeaningsAtIndecies.get_output_meanings()

    for index, decision in output_index_at_decision.items():
        if decision in CONTROLS['decisions_to_fit']:
            print('=' * 120)
            print(f"Decision {decision}")
            model_obj = AIModel.BaseModel(index, CONTROLS['verbose'])

            # Calling correct model functiton
            if CONTROLS['MODEL'] == 0:
                model_obj.single_input_model()
            elif CONTROLS['MODEL'] == 1:
                model_obj.multi_input_model()
            elif CONTROLS['MODEL'] == 2:
                pass

            methods_to_call = CONTROLS['methods_to_call']
            model_obj.compile_model()
            model_obj.train_model()
            if methods_to_call >= 2:
                model_obj.explain_model()
            if methods_to_call >= 3:
                model_obj.evaluate_model()
            if methods_to_call >= 4:
                model_obj.predict_with_model()

            print('=' * 120)

            # Releasing the object references to make the memory eligible
            # for the python garbage memory manager
            print('*' * 20)
            del model_obj
            print('*' * 20)

####################
# Program Controls #
#####################
if __name__ == '__main__':
    '''
    MODEL 0 -> Single Input Model
    MODEL 1 -> Multi Input Model
    MODEL 2 -> Recurent Model (Not built yet)
    -----------------------------------------
    methods_to_call -> Explain
    '''
    # Rename verbose variable to something that makes sense
    main(MODEL=1, decisions_to_fit=['1a'], methods_to_call=1, verbose=1)
