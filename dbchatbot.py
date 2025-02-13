
import json
import os
from langchain.chains import LLMChain
import plotly.graph_objs as go
import plotly.express as px
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain.memory import ConversationBufferMemory
from llama_index.core import VectorStoreIndex
from chromadb.config import Settings as chroma_settings
from langchain.chains import create_sql_query_chain
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from langchain_community.utilities import SQLDatabase
from llama_index.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from llama_index.core import PromptTemplate as PT
import chromadb
import tempfile
import pandas as pd
import sqlite3
from dotenv import load_dotenv


class DatabaseQueryVisualizer:
    def __init__(self, openai_api_key: str = None):
        self.db = None
        self.conn = None
        self.db_type = None
        self.query_result = None
        self.custom_graphs = {}
        self.sql_queries = None
        self.query_outputs = []
        self.context = None
        self.processed = False
        self.db_connected = False
        self.query_engine = None
        self.chroma_client = None
        self.query_doc_result = []

        # Load environment variables
        load_dotenv()

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        # if not self.openai_api_key:
        #     raise ValueError(
        #         "OPENAI_API_KEY not found in environment variables")
        if not self.llama_parse_api_key:
            raise ValueError(
                "LLAMA_PARSE_API_KEY not found in environment variables")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",
                         api_key=self.openai_api_key)

        embed_model = OpenAIEmbedding(
            api_key=self.openai_api_key, model="text-embedding-3-small")
        Settings.embed_model = embed_model
        Settings.llm = llm

    def delete_collection_if_exists(self, collection_name: str):
        """Delete a collection if it exists in the ChromaDB client."""
        if not self.chroma_client:
            raise ValueError(
                "ChromaDB client not initialized. Call parse_file first.")

        existing_collections = self.chroma_client.list_collections()

        for collection in existing_collections:
            if collection.name == collection_name:
                self.chroma_client.delete_collection(collection_name)
                print(f"Collection '{collection_name}' deleted.")
                return
        print(f"Collection '{collection_name}' does not exist.")

    def connect_to_database(self, db_type, **connection_details):
        self.db_type = db_type

        if db_type == "SQLite":
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db_path = temp_db.name
            temp_db.close()

            if 'db_file' in connection_details:
                with open(connection_details['db_file'], 'rb') as src, open(temp_db_path, 'wb') as dst:
                    dst.write(src.read())

            self.conn = sqlite3.connect(temp_db_path, check_same_thread=False)
            self.db = SQLDatabase.from_uri(f"sqlite:///{temp_db_path}")
            self.temp_db_path = temp_db_path
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        if self.db:
            self.context = self.db.get_context()
            self.db_connected = True
            print("context :- ", type(self.context["table_info"]))
            print(f"Connected to {db_type} database.")
        print("----------------------------------------------------------")
        sample_questions = self.questions()
        print("Sample questions generated based on the database schema:")
        # print(sample_questions)
        print("----------------------------------------------------------")
        return sample_questions

    def questions(self):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",
                         api_key=self.openai_api_key)

        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = ans[:5]
        dynamic_template = '\n'.join(dynamic_template)

        template_string = '''{dynamic_template}\

        Only use the following tables and schema\
        {table_info}\

        Your task is to generate 10 sample questions based on the {table_info} for SQL queries and dashboard KPIs.\

        Note: Only provide a list of questions in list format, no explanations needed.
        '''

        prompt = PromptTemplate(
            template=template_string,
            input_variables=['table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

        _input = prompt.format_prompt(table_info=self.context["table_info"])
        output = llm.call_as_llm(_input.to_string())
        # print("Output :- ",output)
        return output

    def parse_file(self, file_paths: list[str]):
        """
        Parse files and create or load a query engine using LlamaIndex.
        For the same set of files, creates a ChromaDB collection only once and reuses it in subsequent calls.

        Args:
            file_paths (list[str]): List of paths to the files to parse
        """
        # Generate a unique collection name based on all file paths
        base_names = "_".join(os.path.basename(path) for path in file_paths)
        collection_name = f'vector_db_{base_names}'
        collection_name = ''.join(
            c if c.isalnum() or c in '-_' else '_' for c in collection_name
        )
        if not collection_name[0].isalpha():
            collection_name = 'db_' + collection_name
        collection_name = collection_name[:63]

        # Initialize ChromaDB client if not already initialized
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=chroma_settings(persist_directory='./db')
            )

        # Check if collection already exists
        collection_exists = False
        chroma_collection = None
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                if collection.name == collection_name:
                    collection_exists = True
                    chroma_collection = collection
                    break
        except Exception as e:
            print(f"Error checking collections: {e}")

        if collection_exists:
            print(f"Loading existing collection: {collection_name}")
            # Load existing collection
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        else:
            print(f"Creating new collection: {collection_name}")
            # Initialize parser and LLM
            parser = LlamaParse(
                result_type="markdown",
                api_key=self.llama_parse_api_key,
                encoding='utf-8'
            )
            self.llm = OpenAI(temperature=0, model="gpt-4o-mini",
                              api_key=self.openai_api_key)

            file_extractor = {
                ".docx": parser,
                ".pdf": parser,
                ".txt": parser,
                ".md": parser,
                "doc": parser
            }

            # Load and parse all files
            documents = SimpleDirectoryReader(
                input_files=file_paths,
                file_extractor=file_extractor
            ).load_data()

            # Embed model setup
            embed_model = OpenAIEmbedding(api_key=self.openai_api_key,model="text-embedding-3-small")
            Settings.embed_model = embed_model
            Settings.llm = self.llm

            # Create new collection
            chroma_collection = self.chroma_client.create_collection(
                name=collection_name
            )

            # Set up vector store
            vector_store = ChromaVectorStore(
                chroma_collection=chroma_collection
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )

            # Create new index
            index = VectorStoreIndex(
                documents,
                storage_context=storage_context
            )

        # Set up query engine
        self.query_engine = index.as_query_engine()
        self.processed = True

        return self.query_engine

    def process_query(self, query, openai_api_key):
        result = {
            "type": None,
            "sql_queries": [],
            "doc_answer": None,
            "query_outputs": [],
            "graphs": [],
        }
        print("Query :- ", query)
        if self.db_connected and not self.processed:
            result["type"] = "database"
            # SQL Query as output
            output = self.run_query(query, openai_api_key)
            print("----------------------------------------------------------")
            print("SQL Query\n")
            print(output)
            print("----------------------------------------------------------")
            result["sql_queries"] = output.split("\n\n")
            result["graphs"] = self.display_results(output)
            result["query_outputs"] = self.query_outputs

        elif self.processed and not self.db_connected:
            result["type"] = "Document"
            doc_answer = self.query_document(query)
            result = doc_answer
            result = str(result)
            print("Answer:")
            print("----------------------------------------------------------")
            print(doc_answer)
            print("----------------------------------------------------------")

        elif self.processed and self.db_connected:
            result["type"] = "both"
            doc_answer = self.query_document(query)
            db_answer = self.run_query(query, openai_api_key)
            df = pd.DataFrame()
            c = self.conn.cursor()
            for query in db_answer.split("\n\n"):
                c.execute(query)
                columns = [description[0] for description in c.description]
                results = c.fetchall()
                df = pd.DataFrame(results, columns=columns)

            from_db = self.compare_answers(
                query, doc_answer, db_answer, openai_api_key, df=df)
            print("**(@!*"*20)
            print(from_db)
            print("**(@!*"*20)
            if from_db:
                print("----------------------------------------------------------")
                print("----------------------------------------------------------")
                result["sql_queries"] = db_answer.split("\n\n")
                result["graphs"] = self.display_results(db_answer)
                result["query_outputs"] = self.query_outputs
            else:
                result["doc_answer"] = doc_answer

                print("----------------------------------------------------------")
                print("doc Answer:")
                print("----------------------------------------------------------")
                print(doc_answer)
                print(type(doc_answer))
                return str(doc_answer)
        print("*&*&*&"*30)
        print(type(result))
        print(result)
        print("*&*&*&"*30)
        return result

    def run_query(self, user_input, openai_api_key):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",
                         openai_api_key=openai_api_key)
        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        template_string = '''
        {dynamic_template}

        Only use the following tables and schema
        {table_info}

        question : {input}

        note : only give SQL query no need to give explanation and output
        If you give more than one sql queries make sure it is seperated by two white spaces, no need to put ```python, ```sql etc. and extra things.
        If there is a Union or and other 2 query join condition like union then dont make a white space before or after on condition.
        If you give more than one sql queries make sure it is seperated by two white spaces and never include counting of sql queries like 1. 2. 3. each query is seprated by two white spaces
        Important Note: For more than one sql queries make sure to seperate it by two white spaces.
        '''

        prompt = PromptTemplate(
            template=template_string,
            input_variables=['input', 'table_info'],
            partial_variables={'dynamic_template': dynamic_template}
        )

        _input = prompt.format_prompt(
            input=user_input, table_info=self.context["table_info"])
        output = llm.call_as_llm(_input.to_string())
        if "```sql" in output:
            output = output.replace("```sql", "").replace("```", "")
        return output

    def generate_graph(self, df, openai_api_key):
        print("GENNNNNNNNNNNN APIIIIIIIII", openai_api_key)

        # Initialize the LLM model (GPT-3.5-Turbo)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",
                         openai_api_key=openai_api_key)

        # SQL query chain creation
        chain = create_sql_query_chain(llm, self.db)

        # Get the first prompt template
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        print("GENNNNNNNNNNNN APIIIIIIIII", openai_api_key)
        print(df.shape)

        # LLM Prompt template for generating Plotly code in JavaScript

        template_string = '''
        You are given the following DataFrame:
        {input}

        Generate a Plotly JavaScript code (using Plotly.newPlot()) to visualize this data in a web browser. Ensure that the graph is visually simple, clear, and accurately represents the data. Avoid creative or complex visualizations—focus on simplicity and clarity so that even non-technical users can easily understand the graph. Return only the JavaScript code for the graph.

        Instructions:
        - Do not use ```javascript``` or any other formatting markers; just return plain JavaScript code.
        - Render the graph inside the HTML element with the id "myDiv" using Plotly.newPlot('myDiv', ...).
        - **Graph Selection**:
            - If the DataFrame has more than one record:
                - Select the graph type based on the column data types (e.g., scatter, line, or bar charts). Avoid defaulting to bar charts unless they are the most appropriate choice for the data.
                - If relationships between columns are unclear or weak, generate individual visualizations for each column instead of forcing a combined graph.
            - If the DataFrame has a **single record with multiple numeric columns**, generate a **simple bar chart** .
            -If the answer has only one single value for (numeric and non-numeric) both output generate a KPI-style visualization using `plotly.graph_objects` ,Avoid complex charts like gauge charts or other charts for single answer value.
            - Avoid combining unrelated columns in a single graph.

         **Handling Non-Numeric Data**:

        - If the DataFrame contains **non-numeric data** such as **dates or text**, do not pass them as the `value` in `go.Indicator`. Instead, display them in the title or as plain text annotations in the visualization.
        - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data like strings or dates, use alternative methods such as titles, text annotations, or other relevant Plotly elements (e.g., `go.Text`)do not include Placeholder for numeric value
        - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data, such as strings or dates, use them in the title or text annotations, but **never in the value**.  


        - **Avoid Complex or Creative Visualizations**:
            - Do not use creative or decorative visualizations like gauge charts, radial charts, or other artistic designs.
            - Keep the visualizations simple, intuitive, and focused on the core data representation.
        - **Error Handling**:
            - Include error handling with a fallback `Plotly.newPlot()` that displays a simple "Graph not displayed" message inside the graph area if rendering fails.
        - **Visualization Guidelines**:
            - Prioritize simplicity and clarity over complexity. Ensure that the graph type and design are visually understandable.
            - Add meaningful titles, axes, and legends to make the graph self-explanatory.
            - Avoid using features that may confuse non-technical users.
        '''

        if df.shape == (1, 1):  # Check if one row and one column
            single_value = df.iloc[0, 0]
            print("Single value:", single_value)

            html_content = f'''
                 <html>
    <body style="margin: 0; height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #f4f4f4;">

        <div id="myDiv" style="width: 300px; height: 150px; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: white; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <p style="font-size: 24px; margin: 0 0 10px 0;">{df.columns.tolist()[0]}</p> <!-- Added bottom margin -->
            <p style="font-size: 24px; margin: 0;">{single_value}</p>
        </div>
    </body>
</html>


            '''

        else:

            # Creating the prompt
            prompt = PromptTemplate(
                template=template_string,
                input_variables=['input'],

            )

            # Generate the prompt with the DataFrame as input
            _input = prompt.format_prompt(input=df)

            # Get the generated JavaScript Plotly code from GPT-3.5 Turbo
            output = llm.call_as_llm(_input.to_string())

            # Embed the generated Plotly JavaScript code into the HTML template

            html_content = f'''
            <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id="myDiv" style="width:100%;height:100%;"></div>
                <script>
                try {{
                    {output}
                }} catch (e) {{
                    // Fallback graph when an error occurs
                    var data = [{{
                        type: "indicator",
                        mode: "number",
                        value: 0,
                        title: {{ text: "Graph not displayed" }}
                    }}];
                    Plotly.newPlot('myDiv', data);
                }}
                </script>
            </body>
            </html>
            '''

        return html_content

    def generate_custom_graph(self, df, x_axis, y_axis, graph_type):
        data = []
        layout = {
            'title': f"{y_axis} vs {x_axis}",
            'xaxis': {'title': x_axis},
            'yaxis': {'title': y_axis}
        }

        if graph_type == "Line":
            trace = {'x': df[x_axis].tolist(), 'y': df[y_axis].tolist(
            ), 'type': 'scatter', 'mode': 'lines'}
        elif graph_type == "Bar":
            trace = {'x': df[x_axis].tolist(
            ), 'y': df[y_axis].tolist(), 'type': 'bar'}
        elif graph_type == "Scatter":
            trace = {'x': df[x_axis].tolist(), 'y': df[y_axis].tolist(
            ), 'type': 'scatter', 'mode': 'markers'}
        elif graph_type == "Pie":
            trace = {'labels': df[x_axis].tolist(
            ), 'values': df[y_axis].tolist(), 'type': 'pie'}
            layout = {'title': f"{y_axis} distribution"}
        data.append(trace)

        # Generate Plotly HTML
        plot_html = f"""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="myDiv" style="width:100%;height:500px;"></div>
        <script>
        try {{
            Plotly.newPlot('myDiv', {json.dumps(data)}, {json.dumps(layout)});
        }} catch (e) {{
            Plotly.newPlot('myDiv', [{{
                type: "indicator",
                mode: "number",
                value: 0,
                title: {{text: "Graph not displayed. Please use Custom Graph."}}
            }}]);
        }}
        </script>
    </body>
    </html>
    """
        return plot_html

    def display_results(self, output):
        c = self.conn.cursor()
        graphs = []
        try:
            for query in output.split("\n\n"):
                print("----------------------------------------------------------")
                print("SQL QUERY")
                print(query)
                c.execute(query)
                columns = [description[0] for description in c.description]
                results = c.fetchall()
                df = pd.DataFrame(results, columns=columns)
                self.query_result = df
                self.query_outputs.append(df)
                print("----------------------------------------------------------")
                print("SQL Output")
                print("----------------------------------------------------------")
                print(df)
                graph_code = self.generate_graph(df, self.openai_api_key)
                if "```python" in graph_code:
                    graph_code = graph_code.replace(
                        "```python", "").replace("```", "")
                print("----------------------------------------------------------")
                print("Generated Graph Code:")
                print(graph_code)
                print("----------------------------------------------------------")
                local_namespace = {"df": df, "go": go}
                # exec(graph_code, local_namespace)
                if 'fig' in local_namespace:
                    graphs.append(local_namespace['fig'])
                print("----------------------------------------------------------")
                print("Custom Graph")
                # x_axis = input(f'Select X-axis from {df.columns}')
                # y_axis = input(f'Select Y-axis from {df.columns}')
                # graph_type = input(
                # "Select from this [\"Line\", \"Bar\", \"Scatter\", \"Pie\"]")
                # custom_graph = self.generate_custom_graph(
                # df, graph_type, x_axis, y_axis)

                graphs.append(graph_code)
        except Exception as e:
            print(f"An error occurred at display results: {e}")
        finally:
            c.close()
        return graphs

    def query_document(self, question: str) -> str:
        """
        Query the parsed document.

        Args:
            question(str): Question to ask about the document

        Returns:
            str: Response from the query engine
        """
        if not self.query_engine:
            raise ValueError("Document not parsed. Call parse_file first.")

        # qa_prompt_temp_str = f"""this is my question {
        #     question} if answer of this question are in this index(vector_database) then only provide me answer othervise say "not provided in file" """
        qa_prompt_temp_str = """
                            "Context information is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Given the context information and not prior knowledge, "
                            "Give answer from the data only and if not present in the data then just say "I'm not equipped to answer this as it falls outside my expertise"."
                            "Query: {query_str}\n"
                            "Answer: "
                            """
        qa_prompt_temp = PT(qa_prompt_temp_str)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_temp})

        response = self.query_engine.query(question)
        print("response:-", response)

        return response

    def chatbot(self, role, openai_api_key):

        self.llm = ChatOpenAI(model="gpt-4o-mini",
                              temperature=0, openai_api_key=openai_api_key)

        # Define the prompt template with the role and history included
        self.prompt_template = PromptTemplate(
            input_variables=["role", "question", "history"],
            template="""
            Please note the following terms and conditions for your responses. this is compulsory!:

            **Content Restrictions**:
                - The chatbot cannot answer any questions related to violence.
                - The chatbot cannot provide answers related to cheating or methods for cheating.
                - The chatbot cannot respond to questions involving sexual content or explicit material.
                - The chatbot cannot engage in or promote harmful, illegal, or unethical behavior.

            **Clarity and Conciseness**:
                - Responses should be clear, concise, and well-structured. Avoid unnecessary jargon or overly complex explanations unless it is relevant to the question.
                - If a question requires a multi-step answer, try to break down the response into clear steps or bullet points.

            **Handling Ambiguity**:
                - If a question is ambiguous or unclear, ask the user for clarification rather than guessing or providing incorrect information.
                - If unsure, the chatbot should respond with: "Can you please clarify your question?" or "I need more information to provide an accurate answer."

            ** Consistency**:
                - Maintain consistent behavior and responses in all interactions. Avoid contradicting previous answers unless there is a valid reason or updated information.

            **Error Handling**:
                - If an error occurs or the chatbot is unable to provide a response, provide a friendly error message, such as: "I'm sorry, I encountered an issue. Can you please rephrase your question?"
                - Avoid showing technical errors or system messages to the user.

            **STRICTLY FOLLOW these instructions ** without deviation.

            You are an expert {role}. You are knowledgeable, insightful, and have extensive experience in this field.
            Please note that you are not permitted to answer questions on topics related to violence, cheating, or any explicit content,
            nor should you provide responses that fall outside your professional role as a {role}.

            Answer questions with professionalism and provide guidance as a skilled {role}. Please respond to the following question:
            [Question]
            Respond in a clear and concise manner, ensuring your answer is relevant to the role of a {role}.

            If the question is not relevant to any of the roles specified, or it pertains to restricted topics, respond with:
            "I'm not equipped to answer this as it falls outside my expertise or contains restricted content."

            Previous conversation:
            {history}

            Question: {question}"""
        )

        # Initialize conversation memory with the correct memory key
        self.memory = ConversationBufferMemory(
            input_key="question",
            memory_key="history",
            return_messages=False
        )

        # Create the conversation chain with the correct configuration
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            memory=self.memory,
            verbose=False
        )

    def ask_question(self, question: str, role: str):
        # Run the conversation chain with inputs and persist memory
        response = self.conversation_chain.predict(
            role=role, question=question)

        print("chatbot answer")
        print(response)
        print("MEMORYYYYYYYYY")
        print(self.memory.load_memory_variables({})["history"])
        return response

    def compare_answers(self, query, doc_answer, db_answer, openai_api_key, df):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                         openai_api_key=openai_api_key)
        prompt_template = """
        You are given two different answers to the following question:

        Question: {query}

        1. doc Answer:
        {doc_answer}

        2. SQL Query Result:
        {db_answer}

        Actual Data from the Database:
        {df}

        Instructions:
        1. Compare both answers to determine which one directly and accurately answers the question.
        2. The doc answer is a natural language response, while the SQL query result may involve technical database details. Choose the answer that is more informative and relevant to the question asked.
        3. If one answer directly answers the question while the other does not (e.g., if one is a SQL query or incomplete information), select the more relevant, human-readable answer.
        4. Return the selected answer exactly as it is , preserving white spaces and formatting.

        Guidelines:
        - If the answer from the SQL Query Database(Answer 2) is more suitable and directly answers the question, return the exact text with '[DB]' appended at the end.
        - If the answer from the document(Answer 1) is more suitable and directly answers the question, return it as it is without any modifications.
        - Focus on selecting the most complete and accurate answer to the specific question, even if one answer includes more technical details that may not directly address the question.
        """
        prompt = PromptTemplate(
            input_variables=["query", "doc_answer", "db_answer", "df"],
            template=prompt_template
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({
            "query": query,
            "doc_answer": doc_answer,
            "db_answer": db_answer,
            "df": df
        })
        print("PDDDDDDDFFFFFFFFFF")
        print(doc_answer)
        print("DBBBBBBBBBBBBBBBBB")
        print(db_answer)
        best_answer = response.strip()
        from_db = best_answer.endswith("[DB]")
        print("FROMMMM DBBBBBBB: ", from_db)
        if from_db:
            best_answer = best_answer.replace("[DB]", "").strip()
        print(best_answer, "!!!!!!!!!!!!!!!!!")
        return from_db


if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = DatabaseQueryVisualizer()
    input_file = ["/Sample_PDF/TechMobile_Smartphone_FAQ.pdf"]

    try:
        # Parse the file
        print("Processing file...")
        visualizer.parse_file(input_file)

        # Optional: Connect to database if needed
        # visualizer.connect_to_database("SQLite", db_file="BIRD.db")
        openai_api_key = os.getenv('OPENAI_API_KEY')
        roles = "Python Developer"
        # visualizer.chatbot(role=roles, openai_api_key=openai_api_key)

        while True:
            # Ask the user for a question or 'exit' to stop
            query = input("Enter your question (type 'exit' to quit): ")

            if query.lower() == "exit":
                print("Exiting the chatbot.")
                break
            result = visualizer.process_query(query, openai_api_key=openai_api_key)
            print(result)
            
            # chat_answer = visualizer.ask_question(
            #     question=question, role=roles)
            # print(chat_answer)

        # query = "what is difference between Yolov4-tiny and Yolov7-tiny ?"
        # result = visualizer.process_query(query, openai_api_key=openai_api_key)
        # print(result)

    except Exception as e:
        print(f"An error occurred at main: {str(e)}")

    finally:
        # Clean up resources if needed
        if hasattr(visualizer, 'conn') and visualizer.conn:
            visualizer.conn.close()
