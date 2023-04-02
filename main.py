from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, \
    ServiceContext
from langchain import OpenAI
import os
import gradio as gr


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(f'text_data/{directory_path}').load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk(f'index_data/{directory_path}_index.json')

    return index


def ask_ai(question: str):
    index = GPTSimpleVectorIndex.load_from_disk(f'index_data/{directory}_index.json')
    response = index.query(question)
    return response.response


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "openai api key 넣을것"

    directory = ""
    select = input("select Directory Path: \n "
                   "1. Fasoo_AnalyticDID \n "
                   "2. Fasoo_Company\n "
                   "3. Fasoo_DataRadar \n "
                   "4. Fasoo_DRM \n "
                   "5. Fasoo_History \n "
                   "6. Fasoo_RiskView \n "
                   "7. Fasoo_Security_Framework \n "
                   "8. Fasoo_Wrapsody \n")
    # construct_index(directory)

    if select == "1":
        directory = "Fasoo_AnalyticDID"
    elif select == "2":
        directory = "Fasoo_Company"
    elif select == "3":
        directory = "Fasoo_DataRadar"
    elif select == "4":
        directory = "Fasoo_DRM"
    elif select == "5":
        directory = "Fasoo_History"
    elif select == "6":
        directory = "Fasoo_RiskView"
    elif select == "7":
        directory = "Fasoo_Security_Framework"
    elif select == "8":
        directory = "Fasoo_Wrapsody"

    iface = gr.Interface(fn=ask_ai, inputs="text", outputs="text")
    iface.launch()
