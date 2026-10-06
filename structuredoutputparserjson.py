from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-26B-A4B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define schema using Pydantic
class Facts(BaseModel):
    fact_1: str
    fact_2: str
    fact_3: str

parser = JsonOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template='Give 3 facts about {topic}\n{format_instructions}',
    input_variables=['topic'],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)

prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
