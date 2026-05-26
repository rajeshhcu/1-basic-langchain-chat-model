from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

information = """
Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum in 1991. Python supports multiple 
programming paradigms including procedural, object-oriented, and functional programming. 
It has a comprehensive standard library and a large ecosystem of third-party packages 
available through PyPI (Python Package Index). Python is widely used in web development, 
data science, artificial intelligence, and automation. Major companies like Google, 
Netflix, and Instagram use Python extensively in their applications.
"""

template = """
First greet the user and then based on the information {information} provided, generate the summary in 2 bullet points.
"""

structured_prompt_template = PromptTemplate(input_variables=["information"],
                                 template=template)

llm = ChatOpenAI()

chain = structured_prompt_template | llm
response = chain.invoke(input={"information": information})
print(response.content)
print(f"Total number of tokens used: {response.response_metadata['token_usage']['total_tokens']}")
