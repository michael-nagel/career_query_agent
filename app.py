# TODO RAG

import os
from typing import Any, Dict, List

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pypdf import PdfReader

# load environment variables from the env file
load_dotenv(override=True)

class Evaluation(BaseModel):
    """Structured format for the Gemini evaluation response."""
    # boolean flag indicating if the response is good
    is_acceptable: bool
    # text feedback explaining the decision
    feedback: str

class Me:
    """Agent class representing Michael Nagel."""

    def __init__(self) -> None:
        """Initialize the agent with personal data and api clients."""
        # initialize the main openai client for generation
        self.openai: OpenAI = OpenAI()
        
        # initialize the gemini client for evaluation using the openai compatibility layer
        self.gemini: OpenAI = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # set the persona name
        self.name: str = "Michael Nagel"
        
        # open the linkedin pdf file
        reader: PdfReader = PdfReader("me/linkedin.pdf")
        
        # initialize an empty string for the linkedin text
        self.linkedin: str = ""
        
        # loop through each page in the pdf
        for page in reader.pages:
            # extract text from the current page
            text: str = page.extract_text()
            
            # append the text if it is not empty
            if text:
                self.linkedin += text
                
        # open the summary text file in read mode
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            # read and store the file contents
            self.summary: str = f.read()

    def system_prompt(self) -> str:
        """Generate the core instructions for the primary agent."""
        # define the base instructions and constraints
        system_prompt: str = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s profile, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions as faithfully as possible. "
            "You are given a summary of his background and LinkedIn profile which you can use to answer questions. "
            "Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            "If you don't know the answer to a question, politely admit that you do not have that information. "
            "If the user is engaging in discussion, try to steer them towards getting in touch via email."
        )

        # append the summary and linkedin data to the prompt
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        
        # add the final behavior reminder
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        
        # hand the compiled prompt back
        return system_prompt

    def get_evaluator_system_prompt(self) -> str:
        """Generate the instructions for the evaluator model."""
        # define the core task for the evaluator
        prompt: str = (
            "You are an evaluator that decides whether a response to a question is acceptable. "
            "You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is of acceptable quality. "
            f"The Agent is playing the role of {self.name} and is representing him. "
            "The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer. "
            f"The Agent has been provided with context on {self.name} in the form of a summary and LinkedIn details."
        )
        
        # append the reference data
        prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        
        # wrap up with the final instruction
        prompt += "With this context, please evaluate the latest response. Reply with whether the response is acceptable and provide your feedback."
        
        # return the full prompt
        return prompt

    def get_evaluator_user_prompt(self, reply: str, message: str, history: List[Dict[str, Any]]) -> str:
        """Format the conversation state for the evaluator."""
        # initialize an empty string for the conversation history
        history_str: str = ""
        
        # format the previous messages into a readable script using the dictionary format
        for msg in history:
            # determine the speaker name
            speaker: str = "User" if msg["role"] == "user" else "Agent"
            
            # extract and append the content
            history_str += f"{speaker}: {msg['content']}\n\n"
            
        # combine all pieces into the final user prompt
        user_prompt: str = f"Here is the previous conversation between the User and the Agent:\n\n{history_str}"
        user_prompt += f"Here is the latest message from the User:\n\n{message}\n\n"
        user_prompt += f"Here is the latest proposed response from the Agent:\n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        
        # return the compiled string
        return user_prompt

    def evaluate(self, reply: str, message: str, history: List[Dict[str, Any]]) -> Evaluation:
        """Use gemini to evaluate the drafted response."""
        # build the message structure for the evaluator
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.get_evaluator_system_prompt()},
            {"role": "user", "content": self.get_evaluator_user_prompt(reply, message, history)}
        ]
        
        # call the gemini api using structured outputs
        response: Any = self.gemini.beta.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=messages,
            response_format=Evaluation
        )
        
        # return the parsed pydantic object
        return response.choices[0].message.parsed

    def rerun(self, original_messages: List[Dict[str, Any]], rejected_reply: str, feedback: str) -> str:
        """Ask the primary agent to try again based on feedback."""
        # append the rejected draft to the context
        original_messages.append({"role": "assistant", "content": rejected_reply})
        
        # append the evaluator feedback as a new user instruction
        correction_prompt: str = (
            f"Your previous response was rejected by the evaluator with the following feedback: {feedback}\n\n"
            "Please provide an updated response that strictly addresses this feedback while staying in character."
        )
        original_messages.append({"role": "user", "content": correction_prompt})
        
        # request a new completion from the primary model
        response: Any = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=original_messages
        )
        
        # return the newly generated text
        return response.choices[0].message.content

    def chat(self, message: str, history: List[Dict[str, Any]]) -> str:
        """Process a user message and return the validated agent response."""
        # build the message history starting with the system prompt
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]
        
        # add the previous conversation history natively since it is already in the exact format we need
        messages.extend(history)
        
        # append the new user message
        messages.append({"role": "user", "content": message})
        
        # request the initial draft completion from the openai api
        response: Any = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
            
        # extract the text response
        reply: str = response.choices[0].message.content
        
        # run the drafted text through the evaluator
        evaluation: Evaluation = self.evaluate(reply, message, history)
        
        # check if the response passed the check
        if evaluation.is_acceptable:
            # log success to the console
            print("Passed evaluation - returning reply")
        else:
            # log the failure and the feedback to the console
            print("Failed evaluation - retrying")
            print(f"Feedback: {evaluation.feedback}")
            
            # generate a new reply using the feedback
            reply = self.rerun(messages, reply, evaluation.feedback)
            
        # return the final validated text
        return reply


# verify if the script is being run directly
if __name__ == "__main__":
    # instantiate the agent object
    me: Me = Me()
    
    # define the initial greeting message
    greeting: str = (
        "Hi! I am an LLM configured to answer questions about Michael based on parsed information from his profile. "
        "To ensure quality, my answers are evaluated by a second LLM before you see them. What would you like to know?"
    )
    
    # create a custom chatbot component initialized with the proper dictionary format
    custom_chatbot: gr.Chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": greeting}]
    )
    
    # launch the gradio chat interface using the custom chatbot component
    gr.ChatInterface(me.chat, chatbot=custom_chatbot).launch()