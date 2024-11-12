import os
import pandas as pd
from langchain.schema import Document  # Import Document class
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_text_splitters import TokenTextSplitter
from langchain_chroma import Chroma

# Load dataset from CSV
dataset_path = "Dataset.csv"  # Update the path to your CSV file
data = pd.read_csv(dataset_path)

# Convert the CSV data into a list of examples
examples = data.to_dict(orient="records")

# Initialize the TokenTextSplitter to split text into manageable chunks
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)

# Process the data into chunks and store it in all_splits
all_splits = []
for example in examples:
    instruction_split = text_splitter.split_text(example["instruction"])
    output_split = text_splitter.split_text(example["output"])
    all_splits.append(instruction_split + output_split)

# Convert the split text into Document objects
documents = []
for split in all_splits:
    for text in split:
        documents.append(Document(page_content=text))

# Initialize HuggingFaceEmbeddings (this handles embedding generation)
embedding_model = HuggingFaceEmbeddings(model_name="microsoft/codebert-base")

# Create a vector store using Chroma and the embedding model
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)

# Define a template for generating PromQL based on user questions
PROMQL_TEMPLATE = """
You are a query assistant. Use the following examples to help create a PromQL query based on the user's natural language request.
If you cannot determine the query, say "I'm not sure."

<examples>
{examples}
</examples>

Question: {question}

Provide the PromQL query only:
"""

# Function to format examples into a string for the prompt
def format_examples(examples):
    return "\n\n".join([f'Instruction: {example["instruction"]}\nOutput: {example["output"]}' for example in examples])

# Create the chain for generating PromQL query
qa_prompt = ChatPromptTemplate.from_template(PROMQL_TEMPLATE)
qa_chain = LLMChain(prompt=qa_prompt, llm=HuggingFaceEmbeddings(model_name="bert-base-uncased"))
# Define a sample question
question = "Show me the average CPU usage over the last hour for all nodes"

# Retrieve relevant documents from the vector store based on the question
docs = vectorstore.similarity_search(question)

# Format the retrieved documents to use as examples in the prompt
formatted_examples = format_examples(docs)

# Run the prompt chain to generate a PromQL query
generated_query = qa_chain.run({"examples": formatted_examples, "question": question})
print("Generated PromQL Query:", generated_query)

# Test the full Q&A retrieval chain with a new question
qa_question = "Get the memory usage over time for all servers"
qa_generated_query = qa_chain.run({"question": qa_question})
print("Generated Q&A PromQL Query:", qa_generated_query)