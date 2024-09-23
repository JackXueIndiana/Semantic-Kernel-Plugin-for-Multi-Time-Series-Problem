import asyncio
from semantic_kernel import Kernel

import pandas as pd
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from injector import Injector, inject, singleton

from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from typing import Tuple
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

class LoggerService:
    def log(self, message: str):
        print(f"LOG: {message}")

class CSVReaderService:
    @inject
    def __init__(self, logger: LoggerService):
        self.logger = logger

    def read_csv(self, file_path: str) -> str:
        self.logger.log(f"Reading CSV file from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.log("CSV file read successfully")
        return df.to_string()
    
    def read_csv_corr(self, file_path: str) -> Tuple[str, float, float, float, float]:
        self.logger.log(f"Reading CSV file from {file_path}")
        df = pd.read_csv(file_path)
        self.logger.log("CSV file read successfully: " + df.to_string())
        
        # Calculate correlation coefficient
        correlation_Temp_pH = df['Temp'].corr(df['pH'])
        correlation_Tr1_pH = df['Tr1'].corr(df['pH'])
        correlation_Tr2_pH = df['Tr2'].corr(df['pH'])
        correlation_Tr3_pH = df['Tr3'].corr(df['pH'])
        print(f"Correlation coefficient between Temperature and pH: {correlation_Temp_pH}")
        print(f"Correlation coefficient between Tr1 and pH: {correlation_Tr1_pH}")
        print(f"Correlation coefficient between Tr2 and pH: {correlation_Tr2_pH}")
        print(f"Correlation coefficient between Tr3 and pH: {correlation_Tr3_pH}")
        
        return df.to_string(), correlation_Temp_pH, correlation_Tr1_pH, correlation_Tr2_pH, correlation_Tr3_pH

class CompletionService:
    @inject
    def __init__(self, api_key: str, endpoint: str, logger: LoggerService):
        self.completion = AzureChatCompletion(
            service_id="Jack Xue",
            deployment_name="gpt-4o",
            api_key="c6670dbd36374a289f8a39a56e1a839c",
            endpoint="https://xjxopenai826.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2023-03-15-preview",
        )
        self.kernel = Kernel()
        self.logger = logger

    
    async def summarize_text(self, df_config: str, df_calibration: str, df_ts: str, correlation_Temp_pH: float, correlation_Tr1_pH: float, correlation_Tr2_pH: float, correlation_Tr3_pH: float) -> str:
        # Create a history of the conversation
        history = ChatHistory(system_message='''
            This is a conversation between an analyst and a scientist. 
            The analyst has read three CSV files and calculated the correlation coefficients between the Temp and pH, Tr1 and pH, Tr2 and pH, Tr3 and pH.
            The sciensiont will summarize the text based on the data provided by the analyst and generate a HTML report.
            Once all the data being collected, generate a HTML report based on the following rules: 
            1 If the last three pH values are lower than LowAlert, and pH highly correlated to the Temp (say correlation efficient is greater than 0.9), and the calibration status is Normal, 
                    Then generate a HTML report that contains a table of 3 rows and 2 columns. 
                    The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Low Temperature'.
                    The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold due to the low temperature.
                    The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH.
                    Remember to include the treatment name and its correlation coefficient with pH in your suggestion.\n\n
            2 Otherwise, generate the same HTML report.
                    The first rows have two cells. One has a string 'Subject' and the other cell has a string 'pH Low ROA: Unknown'.
                    The second row has two cells. One has a string 'Root cause analysis' and in the other cell you state that the system is in Abnormal status and the pH values are lower than the threshold but may not be due to the low temperature.
                    The third row has two cells. One has a string 'Suggested Actions' and in the other cell you suggest to increasing the Treatment Tr1, Tr2 or Tr3 which has the highest correlation with pH.
                    Remember to include the treatment name and its correlation coefficient with pH in your suggestion.
            The HTML report should be in the format of <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">.
            The scientist will generate one HTML report based on the analysis on the data provided by the analyst. The HTML report should be ready to display in a browser.
            The scientist will generate a HTML report using the exact values provided by the analyst.
            The scientist is laser focused on the goal at hand.
            Don't waste time with chit chat.
                              ''')
        history.add_user_message(f'''
            The data provided by the analyst are as follows:\n\n
            1. {df_config} read from a csv file and has four columns: ID, Measure, LowAlarm, UpperAlarm. This file has two rows. The first row is the column names, and the second row is the values.\n\n
            2. {df_calibration} read from a csv file has three columns: ID, Measure, Status. This file has two rows. The first row is the column names, and the second row is the values.\n\n
            3. {df_ts} read from a CSV file. This file has six columns: Time, Temp, Tr1, Tr2, Tr3, pH. This file has 26 rows. The first row is the column names, and the other rows are hourly measurements.\n\n
                Based on the time series, the Pearson correlation coefficients between the Temp and pH has value of {correlation_Temp_pH}\n\n
                Based on the time series, the Pearson correlation coefficients between the Tr1 and pH has value of {correlation_Tr1_pH}\n\n
                Based on the time series, the Pearson correlation coefficients between the Tr2 and pH has value of {correlation_Tr2_pH}\n\n
                Based on the time series, the Pearson correlation coefficients between the Tr3 and pH has value of {correlation_Tr3_pH}.\n\n.
            ''')
        execution_settings = AzureChatPromptExecutionSettings()
        
        self.logger.log("Summarizing text")
        response = await self.completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=self.kernel)
        summary = response
        print(summary)
        self.logger.log("Text summarized successfully")
        return summary
    
@singleton
class AnalystAgent:
    @inject
    def __init__(self, csv_reader: CSVReaderService, logger: LoggerService):
        self.csv_reader = csv_reader
        self.logger = logger

    def read_csv(self, file_path: str) -> str:
        self.logger.log("AnalystAgent: Reading CSV file: " + file_path)
        return self.csv_reader.read_csv(file_path)
    
    def read_csv_corr(self, file_path: str) -> Tuple[str, float, float, float, float]:
        self.logger.log("AnalystAgent: Reading CSV file: " + file_path)
        return self.csv_reader.read_csv_corr(file_path)

@singleton
class ScientistAgent:
    @inject
    def __init__(self, completion: CompletionService, logger: LoggerService):
        self.completion = completion
        self.logger = logger

    def summarize_text(self, df_config: str, df_calibration: str, df_ts: str, correlation_Temp_pH: float, correlation_Tr1_pH: float, correlation_Tr2_pH: float, correlation_Tr3_pH: float) -> str:
        self.logger.log("ScientistAgent: Summarizing text")
        return self.completion.summarize_text(df_config, df_calibration, df_ts, correlation_Temp_pH, correlation_Tr1_pH, correlation_Tr2_pH, correlation_Tr3_pH)

def configure(binder):
    binder.bind(LoggerService, to=LoggerService, scope=singleton)
    binder.bind(CSVReaderService, to=CSVReaderService, scope=singleton)
    binder.bind(CompletionService, to=CompletionService, scope=singleton)
    binder.bind(AnalystAgent, to=AnalystAgent, scope=singleton)
    binder.bind(ScientistAgent, to=ScientistAgent, scope=singleton)

async def main():
    injector = Injector([configure])
    analyst_agent = injector.get(AnalystAgent)
    scientist_agent = injector.get(ScientistAgent)

    df_config = analyst_agent.read_csv('.\\data\\sensor_config.csv')
    df_calibration = analyst_agent.read_csv('.\\data\\sensor_calibration.csv')
    df_ts, correlation_Temp_pH, correlation_Tr1_pH, correlation_Tr2_pH, correlation_Tr3_pH = analyst_agent.read_csv_corr('.\\data\\time_series_data_1.csv')
    
    summary = await scientist_agent.summarize_text(df_config, df_calibration, df_ts, correlation_Temp_pH, correlation_Tr1_pH, correlation_Tr2_pH, correlation_Tr3_pH)
    print(summary)
    return summary

'''
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
'''