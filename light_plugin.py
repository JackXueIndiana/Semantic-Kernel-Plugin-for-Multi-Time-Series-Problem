import asyncio
import logging
from typing import List

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from semantic_kernel.functions import kernel_function

import os
import csv
import pandas as pd
from typing import Annotated

class CSVPlugin:
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    @kernel_function(
        name="read_sensor_config",
        description="read_sensor_config"
    )
    async def read_sensor_config(self) -> Annotated[str, "the output is pd.DataFrame"]:
        file_path = os.path.join(".\data", 'sensor_config.csv')
        
        # Read the CSV file into a DataFrame
        self.df = pd.read_csv(file_path)

        # Display the first few rows of the DataFrame
        self.df.to_json(orient='records')[1:-1].replace('},{', '} {')
        print(self.df)
        return self.df
    
    @kernel_function(
        name="read_sensor_calibration",
        description="read_sensor_calibration"
    )
    async def read_sensor_calibration(self) -> Annotated[str, "the output is pd.DataFrame"]:
        file_path = os.path.join(".\data", 'sensor_calibration.csv')
        
        # Read the CSV file into a DataFrame
        self.df1 = pd.read_csv(file_path)

        # Display the first few rows of the DataFrame
        self.df1.to_json(orient='records')[1:-1].replace('},{', '} {')
        print(self.df1)
        return self.df1
    
    @kernel_function(
        name="read_time_series_data",
        description="read_time_series_data"
    )
    async def read_time_series_data(self) -> Annotated[str, "the output is pd.DataFrame"]:
        file_path = os.path.join(".\data", 'time_series_data_1.csv')
        
        # Read the CSV file into a DataFrame
        self.df2 = pd.read_csv(file_path)

        # Display the first few rows of the DataFrame
        self.df2.to_json(orient='records')[1:-1].replace('},{', '} {')
        print(self.df2)
        
        # Calculate correlation coefficient
        correlation_Temp_pH = self.df2['Temp'].corr(self.df2['pH'])
        correlation_Tr1_pH = self.df2['Tr1'].corr(self.df2['pH'])
        correlation_Tr2_pH = self.df2['Tr2'].corr(self.df2['pH'])
        correlation_Tr3_pH = self.df2['Tr3'].corr(self.df2['pH'])
        print(f"Correlation coefficient between Temperature and pH: {correlation_Temp_pH}")
        print(f"Correlation coefficient between Tr1 and pH: {correlation_Tr1_pH}")
        print(f"Correlation coefficient between Tr2 and pH: {correlation_Tr2_pH}")
        print(f"Correlation coefficient between Tr3 and pH: {correlation_Tr3_pH}")
        
        return self.df2, correlation_Temp_pH, correlation_Tr1_pH, correlation_Tr2_pH, correlation_Tr3_pH
        
async def main():
    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name="gpt-4o",
        api_key="your key",
        endpoint="your endpoint",
    )
    kernel.add_service(chat_completion)

    # Set the logging level for semantic_kernel.kernel to DEBUG.
    setup_logging()
    #logging.setLevel(logging.DEBUG)

    # Add a plugin (the EmailPlugin class is defined above)
    kernel.add_plugin(
        CSVPlugin(),
        plugin_name="CSVPlugin",
    )

    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
    execution_settings.function_call_behavior = FunctionCallBehavior.EnableFunctions(auto_invoke=True, filters={})
    
    # Create a history of the conversation
    history = ChatHistory(system_message="""
        You are a friendly assistant who likes to follow the rules. You will complete required steps
        and request approval before taking any consequential actions. If the user doesn't provide
        enough information for you to complete a task, you will keep asking questions until you have
        enough information to complete the task.
        """)
    history.add_user_message('''
                             1. Use function read_sensor_config read the content from the CSV file from .\data\sensor_config.csv.
                                This file has four columns: ID, Measure, LowAlarm, UpperAlarm.
                                This file has two rows. The first row is the column names, and the second row is the values.
                             2. Use function read_sensor_calibration read the content from the CSV file from .\data\sensor_calibration.csv.
                                This file has three columns: ID, Measure, Status.
                                This file has two rows. The first row is the column names, and the second row is the values.
                             3. Use function time_series_data_1 read the content from the CSV file from .\data\time_series_data_1.csv.
                                This file has six columns: Time, Temp, Tr1, Tr2, Tr3, pH.
                                This file has 26 rows. The first row is the column names, and the other rows are hourly measurements.
                                After reading this function will compute the correlation coefficients between the Temp and pH, Tr1 and pH, Tr2 and pH, and Tr3 and pH from the time series.
                             4. Once all the data being collected, conduct the following analysis: 
                                4.1 If the last three pH values are lower than LowAlert, and pH highly correlated to the Temp (say correlation efficient is greater than 0.9), and the calibration status is Normal, 
                                    Then generate a HTML report that contains a table of 3 rows and 2 columns. 
                                    The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Low Temperature'.
                                    The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold due to the low temperature.
                                    The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH.
                                    Remember to include the treatment name and its correlation coefficient with pH in your suggestion.
                                4.2 Otherwise, generate the same HTML report.
                                    The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Unknown'.
                                    The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold but may not be due to the low temperature.
                                    The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH.
                                    Remember to include the treatment name and its correlation coefficient with pH in your suggestion.
                             P.S. The HTML report should be in the format of <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">.
                             ''')

    result = await chat_completion.get_chat_message_content(
        chat_history=history,
        settings=execution_settings,
        kernel=kernel
    )
    print(result)
    return result
'''
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
'''
