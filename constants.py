import os
from dotenv import load_dotenv

load_dotenv()

QUESTIONS_DT = [
    "Cosa gira dopo Riconciliazione da File?",
    "Cosa gira dopo Riconciliazione da File e dopo quanto tempo?",
    "Fammi un elenco di tutte le sequenze di processi.",
    "Dopo quanto tempo parte il processo SH - Flusso variazioni su nominativo?",
    "Dopo quanto tempo parte il processo Flusso bonifici?",
    "Ho appena lanciato il processo SH - Caricamento FC Forzato, ora cosa devo fare?",
    "Hanno girato i seguenti processi: BAAS - Caricamento Flusso?",
    "Hanno girato i seguenti processi: BAAS - Caricamento Flusso, Caricamento astronavi e Flusso Oppenheimer. Quali processi mancano da girare?",
    "Hanno girato i seguenti processi: BAAS - Caricamento Flusso, Caricamento astronavi. Quali processi mancano da girare?",
]

QUESTIONS = []

QUESTIONS_MULTI_DOC = []

OVERLOAD_SYSTEM_TEMPLATE = """### System : You are an AI assistant that helps people find information using the context provided to answer the question of the User. 
If the answer is not in the context just say that you don't know in a polite way, don't try to make up an answer. 
The text you will find in the context will have all the information you need to answer the question. 
You MUST always include the file, the chapter and the page in which you found the information!
You MUST provide an helpful answer in syntactically correct Italian!"""

# DT_SYSTEM_TEMPLATE = """### Instruction:
# You are an AI assistant that helps people find information using the context provided to answer the question of the User.
# If the answer is not in the context just say that you don't know in a polite way, don't try to make up an answer.
# The text in the context will contain all the information needed to answer the question."process_A" and "process_B" are processes in cascade but limited by the 'id' which they refer.
# Also process_A is always executed before process_B.
# timedelta_mean_minutes is the delta time between the execution of A and B. timedelta_std_minutes is the standard deviation of the delta time.
# timedelta_mean_minutes summed to timedelta_std_minutes is also a good measure that can reply to the questions related to when to launch the process B.
# You MUST provide an helpful answer in syntactically correct Italian!"""

DT_SYSTEM_TEMPLATE = """### Instruction: 
You are an AI assistant that helps provide insights about business processes and timelines using context from data to answer questions from users
in a polite and helpful way, only using information present in the provided context. 
Do not make up answers if the question cannot be answered by using the data.
The context describes the execution timeline for two business processes, process_A and process_B. 
Process_A and process_B run sequentially in a cascade, linked by a common ID. Process_A always finishes executing before process_B begins.
The JSON data provides statistics on the time delay between process_A and process_B over many executions with the timedelta_mean_minutes and the timedelta_std_minutes fields. 
Adding these together gives a useful estimate for when process_B typically begins after process_A.
Respond in grammatically correct Italian.
"""

DT_PERIOD_SYSTEM_TEMPLATE = """### Instruction: 
You are an AI assistant that helps provide insights about business processes using context from data to answer questions from users
in a polite and helpful way, only using information present in the provided context. 
Do not make up answers if the question cannot be answered by using the data.
The context describes the execution of a business process, specified in the 'name' field. 
The 'period' indicates how frequently a process is executed. 
The 'day_of_week' parameter contains the numerical equivalent of a weekday: 0 = lunedì, 1 = martedì and so on.
"""
# COT = """
# Example:

# USER: Cosa gira dopo EAR - Mutui e dopo quanto tempo?
# ASSISTANT: Pensiamo step per step:
# Il "EAR - Mutui" è un "process_A" da come posso notare. Pertanto il "process_B" associato è il processo "Completamento movimenti banca".
# Il processo "Completamento movimenti banca" parte solitamente dopo timedelta_mean_minutes + timedelta_std_minutes minuti. Ossia  146.816666666667 + 62.2166666666667 minuti
# con una probabilità del 0.763636363636364 come indicato nella voce "prob"\n\n

# USER: Cosa gira dopo il processo Marco Cacco e dopo quanto tempo?
# ASSISTANT: Pensiamo step per step:
# Controllo se il processo Marco Cacco è presente nella base dati. Il processo Marco Cacco non esiste pertanto non è possibile determinare cosa giri dopo e con quale probabilità.
# Prova a esplicitare meglio il nome del processo.\n\n
# """

COT = """Example:
USER: Ogni quanto gira il processo BNL per la vita?
ASSISTANT: Pensiamo step per step:
Controllo se è presente il processo BNL per la vita. Ok, l'ho trovato! Il periodo indicato come stringa "B" pertanto so che si riferisce a un 'processo giornaliero' e quindi gira giornalmente. 
"""

STUFF_TEMPLATE = (
    OVERLOAD_SYSTEM_TEMPLATE
    + """

    Context:
    ---------
    {context}
    ---------

    ### USER: {question}.

    ### ASSISTANT:"""
)

DIGITAL_TWIN_TEMPLATE = (
    DT_PERIOD_SYSTEM_TEMPLATE
    + """

    Context:
    ---------
    {}
    ---------
    {}

    {}
    ### Response:"""
)


PARAMS = {
    "embedding_model": os.environ.get("EMBEDDINGS_MODEL_NAME"),
    "target_source_chunks": int(os.environ.get("TARGET_SOURCE_CHUNKS", 4)),
    "qa": [],
    "temperature": 0.0,
    "max_tokens_field": 512,
    "paragraph_overlap": 0,
    "top_k": 50,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
    "chain_type": "stuff",
    "remote_emb": False,
    "remote_model": False,
}
