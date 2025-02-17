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
# from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from langchain_community.utilities import SQLDatabase
from llama_index.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from llama_index.core import PromptTemplate as PT
from llama_index.vector_stores.postgres import PGVectorStore
from psycopg2 import sql
import textwrap
from sqlalchemy import make_url
import psycopg2
import chromadb
import tempfile
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class DatabaseQueryVisualizer:
    def __init__(self):
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
        self.conn_pgadmin=None
        
        # Load environment variables
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0", api_key=self.openai_api_key)
        embed_model = OpenAIEmbedding(api_key=self.openai_api_key,model="text-embedding-3-small")
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.llama_parse_api_key:
            raise ValueError("LLAMA_PARSE_API_KEY not found in environment variables")
        
    def connect_to_database(self, db_type, **connection_details):
        self.db_type = db_type
        
        if db_type == "SQLite":
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db_path = temp_db.name
            print("db name: ",temp_db_path)
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
            self.table_names=self.context['table_names']
            print("All Tables Available: ",self.table_names)
            print(f"Connected to {db_type} database.")
        print("----------------------------------------------------------")
        sample_questions = self.questions()
        print("Sample questions generated based on the database schema:")
        # print(sample_questions)
        print("----------------------------------------------------------")
        return sample_questions
    
    def questions(self):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0", api_key=self.openai_api_key)

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
    
    def delete_pdf(self,pdf_name,conn,table_name):
        table_name='data_'+table_name
        cursor=conn.cursor()
        ids=[]
        #query to fetch all ids with that pdf name
        cursor.execute(f"SELECT id FROM {table_name} WHERE metadata_ ->> 'file_name' = '{pdf_name}';")
        i=cursor.fetchall()
        if not i:
            print("No Such PDF Found")
        else:
            print("--PDF Found--")
            for id in i:
                ids.append(id[0])
            if len(ids)==1:
                ids=f"({ids[0]})"
            else:
                ids=tuple(ids)
            print(ids)
            #deleting all ids
            cursor.execute(f"DELETE FROM {table_name} WHERE id IN {ids};")
            print("deletion done")
            cursor.execute(f"SELECT id from {table_name} where id in {ids};")
            check=cursor.fetchall()
            if not check:
                print("pdf deleted")
                #it deleted table (usecase) if all the pdfs are deleted.
                cursor.execute(f"SELECT metadata_ FROM {table_name};")
                Pdf_exists=cursor.fetchall()
                if not Pdf_exists:
                    print("Deleting Table")
                    cursor.execute(f"DROP TABLE {table_name}")
                    return
            else:
                print("error occured")
            
        #deleted
        print("finish")
    
    def parse_file(self, file_path: list):
        """
        Parse a file and create or load a query engine using LlamaIndex.
        For same PDF, creates ChromaDB collection only first time and reuses it in subsequent calls.
        
        Args:
            file_path (str): Path to the file to parse
        """
        table_exists=False
        table_name='datascience'
        username='postgres'
        password='krishnabarot'
        host='localhost'
        port='5433'
        db_name = "vector_db"
        parser = LlamaParse(
                    result_type="markdown",
                    api_key=self.llama_parse_api_key,
                    encoding='utf-8'
                )

        self.llm = OpenAI(temperature=0, model="gpt-4o-mini", api_key=self.openai_api_key)
        file_extractor = {
                    ".docx": parser,
                    ".pdf": parser,
                    ".txt": parser,
                    ".md": parser,
                    "doc": parser
                }
        embed_model = OpenAIEmbedding(api_key=self.openai_api_key,model='text-embedding-3-small')
        

        # llm2 = Gemini(
        #     model="models/gemini-1.5-flash",
        #     api_key="AIzaSyDPMaJ1biEkINQbXjR6iVuo8tF_VQ62Xr0",  # uses GOOGLE_API_KEY env var by default
        # )
        # embed_model2 = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        
        Settings.embed_model = embed_model
        Settings.llm = self.llm

        conn_str_temp=f"postgresql://{username}:{password}@{host}:{port}"
        self.conn_pgadmin_temp = psycopg2.connect(conn_str_temp)
        self.conn_pgadmin_temp.autocommit=True
        with self.conn_pgadmin_temp.cursor() as c:
            c.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';")
            result=c.fetchone()
            if result:
                print("database exists")
            else:
                print("database did not exists..creating new one")
            
                c.execute(f"CREATE DATABASE {db_name}")
            #     print("db created")
            # print("connected to db "+db_name)

        #PostgreSQL->pgvector
        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
        
        if self.conn_pgadmin is None:
            self.conn_pgadmin = psycopg2.connect(connection_string)
            print("connection done successfully")
            self.conn_pgadmin.autocommit = True

        #new
        cursor = self.conn_pgadmin.cursor()
        cursor.execute(
            sql.SQL("""
                SELECT pid FROM pg_stat_activity
                WHERE datname = %s AND pid <> pg_backend_pid()
            """),
            [db_name]
        )

        active_sessions = cursor.fetchall()

        if active_sessions:
            # Terminate the active sessions
            for session in active_sessions:
                pid = session[0]
                print(f"Terminating session with PID {pid}...")
                cursor.execute(
                    sql.SQL("SELECT pg_terminate_backend(%s)"),
                    [pid]
                )
            print("All active sessions terminated.")
        else:
            print("No active sessions found.")

        
        # cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", ('your_database_name',))
        # result = cursor.fetchone()

        with self.conn_pgadmin.cursor() as c:
            c.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';")
            result=c.fetchone()
            if result:
                print("database exists")
            else:
                print("database did not exists..creating new one")
            
                c.execute(f"CREATE DATABASE {db_name}")
                print("db created")
            print("connected to db "+db_name)
            

            cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables=cursor.fetchall()
            if tables:
                for table in tables:
                    name=table[0]
                    # print(table[0])
                    # print("----")
                    if name=='data_'+table_name:
                        table_exists=True
                        break
                        

            if table_exists:
                print(f"Using already existing table: {table_name}")
                new_docs=[]
                all_files=[]
                cursor.execute(f"SELECT DISTINCT metadata_->>'file_name' AS file_name FROM data_{table_name};")
                files=cursor.fetchall()
                
                for file in files:
                    all_files.append(file[0])
                # print("all pdf names: ",all_files)

                for file in file_path:
                    file_name=os.path.basename(file)
                    if file_name not in all_files:
                        # print("--Found new file--",file_name)
                        new_docs.append(file)
                    else:
                        print("File is already there in db ",file_name)
                # print("new docs: ",new_docs)

                
                url = make_url(connection_string)
                vector_store = PGVectorStore.from_params(
                                database=db_name,
                                host=url.host,
                                password=url.password,
                                port=url.port,
                                user=url.username,
                                table_name=table_name,
                                embed_dim=1536,  # openai embedding dimension 10104,26768
                                    )
                
                if new_docs:
                    doc=SimpleDirectoryReader(input_files=new_docs, 
                            file_extractor=file_extractor).load_data()
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    # print("new pdfs added ")

                    # Create new index
                    index = VectorStoreIndex(doc, storage_context=storage_context)
                    self.query_engine = index.as_query_engine()
                else:
                    # print("all same pdfs")
                    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                    self.query_engine = index.as_query_engine()
                    # print(f"loaded from existing data_{table_name}")
                self.processed = True
                # a=input("press ok for deleting ")
                # if a=='ok':
                #     self.delete_pdf("sample5.txt",self.conn_pgadmin,table_name)
            else:
                # Load and parse documents
                documents = SimpleDirectoryReader(
                            input_files=file_path, 
                            file_extractor=file_extractor
                        ).load_data()
                
                    

                url = make_url(connection_string)
                vector_store = PGVectorStore.from_params(
                    database=db_name,
                    host=url.host,
                    password=url.password,
                    port=url.port,
                    user=url.username,
                    table_name=table_name,
                    embed_dim=1536,  # openai embedding dimension 10104,26768
                )

                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                print("table created ")

                # Create new index
                index = VectorStoreIndex(documents, storage_context=storage_context)
                self.query_engine = index.as_query_engine()
                self.processed = True
                

        return self.query_engine
        #end of pgvector


        # for file_path in file_paths:
        #     # Generate a unique collection name based on file path
        #     base_name = os.path.basename(file_path)
        #     collection_name = f'vector_db_{os.path.splitext(base_name)[0]}'
        #     collection_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in collection_name)
        #     if not collection_name[0].isalpha():
        #         collection_name = 'db_' + collection_name
        #     collection_name = collection_name[:63]

        #     # Initialize ChromaDB client if not initialized or is None
        #     if self.chroma_client is None:
        #         self.chroma_client = chromadb.PersistentClient(
        #             path="./chroma_db",
        #             settings=chroma_settings(persist_directory='./db')
        #         )

        #     # Check if collection already exists
        #     collection_exists = False
        #     try:
        #         collections = self.chroma_client.list_collections()
        #         for collection in collections:
        #             if collection.name == collection_name:
        #                 collection_exists = True
        #                 chroma_collection = collection
        #                 break
        #     except Exception as e:
        #         print(f"Error checking collections: {e}")
        #         collection_exists = False

        #     if collection_exists:
        #         print(f"Loading existing collection: {collection_name}")
        #         # Load existing collection
        #         vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        #         storage_context = StorageContext.from_defaults(vector_store=vector_store)

        #         # Load existing index
        #         index = VectorStoreIndex.from_vector_store(
        #             vector_store=vector_store,
        #             storage_context=storage_context
        #         )
        #     else:
        #         print(f"Creating new collection: {collection_name}")

        #         # Initialize parser and LLM
        #         parser = LlamaParse(
        #             result_type="markdown",
        #             api_key=self.llama_parse_api_key,
        #             encoding='utf-8'
        #         )

        #         self.llm = OpenAI(temperature=0, model="gpt-4o-mini", api_key=self.openai_api_key)

        #         file_extractor = {
        #             ".docx": parser,
        #             ".pdf": parser,
        #             ".txt": parser,
        #             ".md": parser,
        #             "doc": parser
        #         }

        #         # Load and parse documents
        #         documents = SimpleDirectoryReader(
        #             input_files=[file_path], 
        #             file_extractor=file_extractor
        #         ).load_data()

        #         embed_model = OpenAIEmbedding(api_key=self.openai_api_key)
        #         Settings.embed_model = embed_model
        #         Settings.llm = self.llm

        #         # Create new collection
        #         chroma_collection = self.chroma_client.create_collection(name=collection_name)

        #         # Set up vector store
        #         vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        #         storage_context = StorageContext.from_defaults(vector_store=vector_store)

        #         # Create new index
        #         index = VectorStoreIndex(documents, storage_context=storage_context)

        #     # Set up query engine
        #     self.query_engine = index.as_query_engine()
        #     self.processed = True

        
    
    def process_query(self, query, openai_api_key):
        result = {
            "type": None,
            "sql_queries": [],
            "doc_answer": None,
            "query_outputs": [],
            "graphs": [],
        }
        print("Query :- " , query)
        if self.db_connected and not self.processed:
            result["type"] = "database"
            #SQL Query as output
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
                
            best_answer, from_db = self.compare_answers(query, doc_answer, db_answer, openai_api_key, df=df)
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
                print("----------------------------------------------------------")
        
        return result
    
    def run_query(self, user_input, openai_api_key):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",openai_api_key=openai_api_key)
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

        _input = prompt.format_prompt(input=user_input, table_info=self.context["table_info"])
        output = llm.call_as_llm(_input.to_string())
        if "```sql" in output:
            output = output.replace("```sql", "").replace("```", "")
        return output
    
    def generate_graph(self, df, openai_api_key):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature="0",openai_api_key=openai_api_key)
        chain = create_sql_query_chain(llm, self.db)
        k = chain.get_prompts()[0]
        ans = k.template.split("\n")
        dynamic_template = '\n'.join(ans[:5])

        template_string = '''
        You are given the following DataFrame:
        {input}

        Generate a Plotly code to visualize this data. Ensure that the graph accurately represents the data and is informative. Only provide the code for the graph.

        - If the DataFrame has more than one record, select the graph type based on the data columns (such as scatter, line, or bar charts). Avoid defaulting to a bar chart unless it's the most appropriate type.
        - If the DataFrame has only one record with **multiple columns**, generate a KPI-style card using `plotly.graph_objects`. The card must display key **numeric values** from the DataFrame. If a column contains non-numeric data (e.g., text, dates), include it in the title or as a text annotation, but do not use it as the value for the KPI.
        - If the DataFrame contains **non-numeric data** such as **dates or text**, do not pass them as the `value` in `go.Indicator`. Instead, display them in the title or as plain text annotations in the visualization.
        - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data like strings or dates, use alternative methods such as titles, text annotations, or other relevant Plotly elements (e.g., `go.Text`).
        - The DataFrame is already imported as 'df', so do not include import statements or redundant definitions.
        - If a column contains non-numeric data (e.g., text, dates), include it in the title or as a text annotation, but **do not use it as the value** for the KPI.
        - Ensure that only **numeric values** are passed to `go.Indicator` as the `value` property. For non-numeric data, such as strings or dates, use them in the title or text annotations, but **never in the value**.  
        Return only the Plotly code for the graph, and No need to put ```python and extra things. also define all lib with code.
        - Always include error handling to prevent crashes use try catch and in catch display a **card** written Graph not displayed. Please use Custom Graph.
        '''

        prompt = PromptTemplate(
            template=template_string,
            input_variables=['input'],
        )

        _input = prompt.format_prompt(input=df)
        output = llm.call_as_llm(_input.to_string())
        return output
    
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
                graph_code = self.generate_graph(df, openai_api_key)
                if "```python" in graph_code:
                    graph_code = graph_code.replace("```python", "").replace("```", "")
                print("----------------------------------------------------------")
                print("Generated Graph Code:")
                print(graph_code)
                print("----------------------------------------------------------")
                local_namespace = {"df": df, "go": go}
                exec(graph_code, local_namespace)
                if 'fig' in local_namespace:
                    graphs.append(local_namespace['fig'])
                print("----------------------------------------------------------")
                print("Custom Graph")
                x_axis = input(f'Select X-axis from {df.columns}')
                y_axis = input(f'Select Y-axis from {df.columns}')
                graph_type = input("Select from this [\"Line\", \"Bar\", \"Scatter\", \"Pie\"]")
                custom_graph = self.generate_custom_graph(df, graph_type, x_axis, y_axis)
                
                graphs.append(custom_graph)
        except Exception as e:
            print(f"An error occurred at display results: {e}")
        finally:
            c.close()
        return graphs
        
    def query_document(self, question: str) -> str:
        """
        Query the parsed document.
        
        Args:
            question (str): Question to ask about the document
            
        Returns:
            str: Response from the query engine
        """
        if not self.query_engine:
            raise ValueError("Document not parsed. Call parse_file first.")
        
        # qa_prompt_temp_str = f"""Question: {question} if answer of this question are in this index(vector_database) then only provide me answer othervise say "not provided in file" """
        qa_prompt_temp_str="""
                            **Strictly give answer from the context provided below and follow rules**
                            "Context information is below.\n"
                            "---------------------\n"
                            "{context_str}\n"
                            "---------------------\n"
                            "Rules: "
                                -"**Use only the information from the provided documents to answer the query.** "\n"
                                -"**Do not generate or infer information outside of the context. If the information is not present in the documents, reply with "Out of context.**"
                                -"**If it is an empty string then say please provide question **"
                                -"**Avoid any form of hallucination or assumptions. Provide answers that are directly supported by the context.**"
                            
                            *Here is the query*
                            "Query: {query_str}\n"
                            "Answer: "                           
                            """
        # print("RAW PROMPT: ",qa_prompt_temp_str)
        # print()
        qa_prompt_temp = PT(qa_prompt_temp_str)
        # print("QA_PROMPT_TEMPLte: ",qa_prompt_temp)
        self.query_engine.update_prompts({"response_synthesizer:text_qa_templete":qa_prompt_temp})
        
        response = self.query_engine.query(question)
        print("response:-",response)
    
        return response
    
    def chatbot(self,role, openai_api_key):

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
        
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

            **Consistency**:
                - Maintain consistent behavior and responses in all interactions. Avoid contradicting previous answers unless there is a valid reason or updated information.

            **Error Handling**:
                - If an error occurs or the chatbot is unable to provide a response, provide a friendly error message, such as: "I'm sorry, I encountered an issue. Can you please rephrase your question?"
                - Avoid showing technical errors or system messages to the user.

            **STRICTLY FOLLOW these instructions** without deviation.

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
        response = self.conversation_chain.predict(role=role, question=question)
        
        print("chatbot answer")
        print(response)
        print("MEMORYYYYYYYYY")
        print(self.memory.load_memory_variables({})["history"])
        return response
    
    def compare_answers(self, query, doc_answer, db_answer, openai_api_key,df):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,openai_api_key=openai_api_key)
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
        4. Return the selected answer exactly as it is, preserving white spaces and formatting.

        Guidelines:
        - If the answer from the SQL Query Database (Answer 2) is more suitable and directly answers the question, return the exact text with '[DB]' appended at the end.
        - If the answer from the document (Answer 1) is more suitable and directly answers the question, return it as it is without any modifications.
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
            "df":df
        })
        print("PDDDDDDDFFFFFFFFFF")
        print(doc_answer)
        print("DBBBBBBBBBBBBBBBBB")
        print(db_answer)
        best_answer = response.strip()
        from_db = best_answer.endswith("[DB]")
        print("FROMMMM DBBBBBBB: ",from_db)
        if from_db:
            best_answer = best_answer.replace("[DB]", "").strip()
        return best_answer
    
if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = DatabaseQueryVisualizer()
    input_file = ["D:\\Ridgeant_Learning\\Docs\\CleanBot_Robotic_Vacuum_Cleaner_FAQ.pdf","D:\\Ridgeant_Learning\\Docs\\sample5.txt","D:\\Ridgeant_Learning\\Docs\\sample_pdf-2.pdf"]
    
    try:
        # Parse the file
        print("Processing file...")
        visualizer.parse_file(input_file)
        
        # # Optional: Connect to database if needed
        # visualizer.connect_to_database("SQLite", db_file="BIRD.db")
        openai_api_key = self.openai_api_key
        roles = "Python Developer"
        # visualizer.chatbot(role=roles, openai_api_key=openai_api_key)
        
        # while True:
        #     # Ask the user for a question or 'exit' to stop
        #     question = input("Enter your question (type 'exit' to quit): ")

        #     if question.lower() == "exit":
        #         print("Exiting the chatbot.")
        #         break
        #     chat_answer = visualizer.ask_question(question=question, role=roles)
        #     print(chat_answer)

        while True:
            # Ask the user for a question or 'exit' to stop
            query = input("Enter your question (type 'exit' to quit): ")

            if query.lower() == "exit":
                print("Exiting the chatbot.")
                break
            if not query:
                    print("")
                    result='please provide question'
            else:
                result = visualizer.process_query(query, openai_api_key=openai_api_key)
                result=str(result)
                print("result type: ",type(result))
            print(result)
            
        # query = "Extended efficient layer aggregation networks"
        # result = visualizer.process_query(query, openai_api_key=openai_api_key)
        # print(result)
        
    except Exception as e:
        print(f"An error occurred at main: {str(e)}")
    
    finally:
        # Clean up resources if needed
        if hasattr(visualizer, 'conn') and visualizer.conn:
            visualizer.conn.close()