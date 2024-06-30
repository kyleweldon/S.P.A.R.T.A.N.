#############################################
# File created by Kyle Weldon on 04/05/2024 #
#############################################

import AIModel
import MeaningsAtIndecies

################### Questions/Talking points for next meeting ##########################
'''
- Why do some people have a 0 as their decision instead of a 1 or 2? 
'''
########################################################################################
################# Needs Doing list for entire project ##############
# TODO: Change the methods_to_call perameter into a list and call all methods in list
# TODO: Change the numpy arrays to be tensors and fix the shape and stuff
# TODO: Fix up multi Input data model
#############################################################

def main(**CONTROLS):
    output_index_at_decision = MeaningsAtIndecies.get_output_meanings()

    for index, decision in output_index_at_decision.items():
        if decision in CONTROLS['decisions_to_fit']:
            print('=' * 120)
            print(f"Decision {decision}")
            model_obj = AIModel.BaseModel(decision_index=index, verbose=CONTROLS['verbose'])

            # Calling correct model functiton
            if CONTROLS['MODEL'] == 0:
                model_obj.single_input_model()
            elif CONTROLS['MODEL'] == 1:
                model_obj.multi_input_model(activiations=['relu', 'relu', 'relu', 'relu'],
                                  units=[80, 64, 32, 40])


            if 1 in CONTROLS['methods_to_call']:
                model_obj.compile_model()
                model_obj.train_model()
            if 2 in CONTROLS['methods_to_call']:
                model_obj.explain_model()
            if 3 in CONTROLS['methods_to_call']:
                model_obj.evaluate_model()
            if 4 in CONTROLS['methods_to_call']:
                model_obj.predict_with_model()

            print('=' * 120)

            # Releasing the object references to make the memory eligible
            # for the python garbage memory manager
            print('*' * 20)
            del model_obj
            print('*' * 20)

def hyperperamiterize_brute(**CONTROLS):

    activation_functions = ['elu',
                            'gelu',
                            'relu',
                            'selu',
                            'softmax',
                            'softplus',
                            'softsign',
                            'swish',
                            'tanh']

    best_accuracy = 0.0
    start_unit = 10
    best_acts = []
    best_units = []
    length = len(activation_functions)
    for num1 in range(length):
        u1 = int((num1 + 1) * start_unit)
        act1 = activation_functions[num1]
        for num2 in range(length):
            u2 = int((num2 + 1) * start_unit)
            act2 = activation_functions[num2]
            for num3 in range(length):
                u3 = int((num3 + 1) * start_unit)
                act3 = activation_functions[num3]
                for num4 in range(length):
                    u4 = int((num4 + 1) * start_unit)
                    act4 = activation_functions[num4]
                    acts = [act1, act2, act3, act4]
                    units = [u1, u2, u3, u4]
                    model = AIModel.BaseModel(verbose=0)
                    model.multi_input_model(activiations=[act1, act2, act3, act4],
                                            units=[u1, u2, u3, u4])
                    model.compile_model()
                    accuracy = model.train_model() * 100
                    del model

                    print('Current:')
                    print(f"\tCurrent accuracy {accuracy}")
                    print(f"\tCurrent units {units}")
                    print(f"\tCurrent activation functitons {acts}")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_acts = acts
                        best_units = units

                    print('Best:')
                    print(f"\tBest accuracy {best_accuracy}")
                    print(f"\tBest units {best_units}")
                    print(f"\tBest activation functitons {best_acts}")

    print('\n'*5)
    print('Dysplaying the best performing model')
    print(best_accuracy)
    print(acts)
    print(units)

####################
# Program Controls #
#####################
if __name__ == '__main__':
    '''
    MODEL 0 -> Single Input Model
    MODEL 1 -> Multi Input Model
    -----------------------------------------
    methods_to_call -> Explain
    '''
    obj = AIModel.BaseModel(decision_index=4, verbose=0)
    obj.clustering()
    # hyperperamiterize_func = False
    # DTF = ['3a']
    # MTC = [1]
    # if hyperperamiterize_func:
    #     hyperperamiterize_brute()
    # else:
    #     main(MODEL=0, decisions_to_fit=DTF, methods_to_call=MTC, verbose=1)



