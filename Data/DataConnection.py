#############################################
# File created by Kyle Weldon on 04/06/2024 #
#############################################

import openpyxl
import numpy as np
np.set_printoptions(threshold=np.inf)


class DataForTraining:
    ###################################################
    # Useful variable names to used in model training #
    ###################################################
    '''
    self.sequential_x_validate
    self.sequential_x_train

    self.big5_x_validate
    self.big5_x_train
    self.layer2_x_validate
    self.layer2_x_train
    self.layer3_x_validate
    self.layer3_x_train
    self.layer4_x_validate
    self.layer4_x_train

    self.y_validate
    self.y_train
    '''
    def __init__(self, decision_index):
        self.decision_index = decision_index
        self.data_for_model_input = 'C:/KyleWeldon/Projects/ThinkTank/S.P.A.R.T.A.N/Data/InputData.txt'
        self.data_for_model_output = 'C:/KyleWeldon/Projects/ThinkTank/S.P.A.R.T.A.N/Data/OutputData.txt'
        self.sequential_inputs = []
        self.input_layer1 = []
        self.input_layer2 = []
        self.input_layer3 = []
        self.input_layer4 = []
        self.output_training_data = []

        with open(self.data_for_model_input, 'r') as input_file, open(self.data_for_model_output, 'r') as output_file:
            input_file_lines = input_file.readlines()
            output_file_lines = output_file.readlines()

            for i in range(228):  # There are 229 lines in both files
                # Dealing with input data
                line = input_file_lines[i].strip().split(',')
                total_inputs = np.array([int(val) for val in line])
                temp_input1 = total_inputs[:5]
                temp_input2 = total_inputs[5:10]
                temp_input3 = total_inputs[10:13]
                temp_input4 = total_inputs[13:24]

                # Dealing with output data
                line = output_file_lines[i].strip().split(',')
                decision = int(line[self.decision_index])
                temp_output = np.array([1 if decision == 2 else 0])

                self.sequential_inputs.append(total_inputs)
                self.input_layer1.append(temp_input1)
                self.input_layer2.append(temp_input2)
                self.input_layer3.append(temp_input3)
                self.input_layer4.append(temp_input4)
                self.output_training_data.append(temp_output)



        # Convert lists to numpy arrays
        self.sequential_x_train = np.array(self.sequential_inputs)
        self.big5_x_train = np.array(self.input_layer1)
        self.layer2_x_train = np.array(self.input_layer2)
        self.layer3_x_train = np.array(self.input_layer3)
        self.layer4_x_train = np.array(self.input_layer4)
        self.y_train = np.array(self.output_training_data)

        # Split data into training data and validation data
        self.sequential_x_validate = self.sequential_x_train[-28:]
        self.sequential_x_train = self.sequential_x_train[:-28]
        self.big5_x_validate = self.big5_x_train[-28:]
        self.big5_x_train = self.big5_x_train[:-28]
        self.layer2_x_validate = self.layer2_x_train[-28:]
        self.layer2_x_train = self.layer2_x_train[:-28]
        self.layer3_x_validate = self.layer3_x_train[-28:]
        self.layer3_x_train = self.layer3_x_train[:-28]
        self.layer4_x_validate = self.layer4_x_train[-28:]
        self.layer4_x_train = self.layer4_x_train[:-28]
        self.y_validate = self.y_train[-28:]
        self.y_train = self.y_train[:-28]


    def __del__(self):
        print('Object released.')

class ExcelConnection:
    def __init__(self):
        self.excel_file_path = "C:/KyleWeldon/Projects/ThinkTank/S.P.A.R.T.A.N/Data/Data.xlsx"
        self.data = []

        # Gather data from the Excel file
        workbook = openpyxl.load_workbook(self.excel_file_path)
        sheet = workbook.active

        for row in sheet.iter_rows(values_only=True):
            self.data.append(row)

        if not self.data:
            print('No data obtained from excel file')
            exit(1)



    def write_data_to_file_from_excel(self):
        input_file = open(self.Data_for_model_input, 'w')
        output_file = open(self.Data_for_model_output, 'w')
        length = len(self.data)

        for i in range(length - 1): # For skipping the first row
            row = self.data[i + 1] # For skipping the first row

            # Index 0 is the person number (not important)
            if int(row[0]) != 232: # Person 232 missing data so ignoring
                # Input data
                input_temp = str(row[1]) + ','
                input_temp += str(row[2]) + ','
                input_temp += str(row[3]) + ','
                input_temp += str(row[4]) + ','
                input_temp += str(row[5]) + ','
                # Skip index 6
                input_temp += str(row[7]) + ','
                input_temp += str(row[8]) + ','
                input_temp += str(row[9]) + ','
                input_temp += str(row[10]) + ','
                input_temp += str(row[11]) + ','
                # Skip index 12
                input_temp += str(row[13]) + ','
                input_temp += str(row[14]) + ','
                input_temp += str(row[15]) + ','
                # Skip index 16
                input_temp += str(row[17]) + ','
                input_temp += str(row[18]) + ','
                input_temp += str(row[19]) + ','
                input_temp += str(row[20]) + ','
                input_temp += str(row[21]) + ','
                input_temp += str(row[22]) + ','
                input_temp += str(row[23]) + ','
                input_temp += str(row[24]) + ','
                input_temp += str(row[25]) + ','
                input_temp += str(row[26]) + ','
                input_temp += str(row[27])
                # Output data
                output_temp = str(row[28])  + ','
                output_temp += str(row[29]) + ','
                output_temp += str(row[30]) + ','
                output_temp += str(row[31]) + ','
                output_temp += str(row[32]) + ','
                output_temp += str(row[33]) + ','
                output_temp += str(row[34]) + ','
                output_temp += str(row[35]) + ','
                output_temp += str(row[36]) + ','
                output_temp += str(row[37]) + ','
                output_temp += str(row[38]) + ','
                output_temp += str(row[39]) + ','
                output_temp += str(row[40]) + ','
                output_temp += str(row[41]) + ','
                output_temp += str(row[42]) + ','
                output_temp += str(row[43]) + ','
                output_temp += str(row[44]) + ','
                output_temp += str(row[45]) + ','
                output_temp += str(row[46]) + ','
                output_temp += str(row[47]) + ','
                output_temp += str(row[48]) + ','
                output_temp += str(row[49])

                input_file.write(f"{input_temp}\n")
                output_file.write(f"{output_temp}\n")
        input_file.close()
        output_file.close()

if __name__ == '__main__':
    obj = DataForTraining(0)
    print('Layer 1:')
    print(obj.big5_x_train[:5])
    print('Layer 2:')
    print(obj.layer2_x_train[:5])
    print('Layer 3:')
    print(obj.layer3_x_train[:5])
    print('Layer 4:')
    print(obj.layer4_x_train[:5])
    print('Output later:')
    print(obj.y_train[:5])
