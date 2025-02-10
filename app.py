import json
import logging
import re
import time
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.debug("Iniciando proyecto")

import gradio as gr
import pytesseract
import vertexai
from google.cloud import storage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field

storage_client = storage.Client()
logging.info("Conectado con Cloud Storage")

BUCKET_NAME = "bucket-grupo4-cxb"
file_name = ""

log_messages = []

def upload_txt_to_gpc(text, output_filename):
    """Guarda el texto extraído en un archivo .txt en la carpeta especificada."""
    txt_path = f"textos/{file_name}/{output_filename}"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(txt_path)

    # Subir TXT a GCS
    blob.upload_from_string(text, content_type="text/plain")
    logging.info(f"Texto guardado: {BUCKET_NAME}/{txt_path}")

def upload_pdf_to_gpc(pdf_bytes):
    """Guarda el PDF en la carpeta especificada."""
    pdf_path = f"archivos/{file_name}.pdf"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(pdf_path)

    # Subir PDF a GCS
    blob.upload_from_string(pdf_bytes, content_type="application/pdf")
    logging.info(f"Archivo guardado: {BUCKET_NAME}/{pdf_path}")

def extract_json(text):
    """
    Extrae JSON desde una cadena de texto usando regex.
    Si la respuesta contiene más texto antes o después del JSON, se filtra.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return match.group()  # Intenta cargar el JSON
        except json.JSONDecodeError:
            return {"error": "No se pudo decodificar JSON"}
    return {"error": "No se encontró JSON en la respuesta"}

def upload_json_to_gpc(json_string):
    """Guarda el texto extraído en un archivo .txt en la carpeta especificada."""
    data = json.loads(json_string)
    json_data = json.dumps(data, indent=4, ensure_ascii=False)

    json_path = f"json_output/{file_name}.json"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(json_path)

    # Subir contenido de texto a GCS
    blob.upload_from_string(json_data, content_type="application/json")
    logging.info(f"JSON guardado: {BUCKET_NAME}/{json_path}")

import fitz  # noqa: I001
def is_searchable_pdf(pdf_bytes):
    """Verifica si el PDF contiene texto digital o es un escaneo."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        if page.get_text().strip():  # Si hay texto real
            doc.close()
            return True
    doc.close()
    return False


def extract_text_from_searchable_pdf(pdf_bytes):
    """Extrae texto de un PDF digitalizado (no escaneado)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_text = []

    logging.info(f"Procesando el arhivo {file_name}")

    for i, page in enumerate(doc):
        logging.info(f"Procesando página {i + 1} (OCR)...")
        text = page.get_text()
        extracted_text.append(text)

        upload_txt_to_gpc(text, f"pagina_{i + 1}.txt")

    full_text = "\n".join(extracted_text)
    upload_txt_to_gpc(full_text, "completo.txt")
    doc.close()
    return full_text


def extract_text_from_scanned_pdf(pdf_bytes):
    """Extrae texto de un PDF escaneado usando OCR."""
    pages = convert_from_bytes(pdf_bytes, 300)  # Convertir PDF en imágenes
    extracted_text = []

    logging.info(f"Procesando el arhivo {file_name}")

    for i, page in enumerate(pages):
        logging.info(f"Procesando página {i + 1} (OCR)...")
        text = pytesseract.image_to_string(page)  # Extraer texto
        extracted_text.append(text)

        upload_txt_to_gpc(text, f"pagina_{i + 1}.txt")

    full_text = "\n".join(extracted_text)
    upload_txt_to_gpc(full_text, "completo.txt")
    return full_text


def extract_text_from_pdf(pdf_file):
    """Descarga el PDF de GCS y extrae su texto, ya sea digitalizado o escaneado."""
    global file_name  # noqa: PLW0603
    file_name=pdf_file.rsplit("/", 1)[-1].replace(".pdf", "")
    with open(pdf_file, "rb") as file:  # noqa: PTH123
        pdf_bytes = file.read()
    logging.info(f"Se ha leido el PDF - {file_name}")

    upload_pdf_to_gpc(pdf_bytes)

    if is_searchable_pdf(pdf_bytes):
        logging.info("PDF con texto digitalizado detectado.")
        return extract_text_from_searchable_pdf(pdf_bytes)
    else:  # noqa: RET505
        logging.info("PDF escaneado detectado, aplicando OCR...")
        return extract_text_from_scanned_pdf(pdf_bytes)


PROJECT_ID = "vegrtll-sot-ia-gcp-08"
REGION = "us-central1"

logging.info("Iniciando VertexAI")
vertexai.init(project=PROJECT_ID, location=REGION)
logging.info("VertexAI Iniciado")

# Configuramos el LLM
llm_trabajador = ChatVertexAI(
    model_name="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=4096,
)

llm_supervisor = ChatVertexAI(
    model_name="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=1024
)

# Prompts de los agentes
system_prompt_agente_extractor = """
Eres un experto en análisis documental y extracción de datos de notas simples de viviendas. Tu tarea es analizar la siguiente nota simple y extraer los datos clave de manera estructurada.
Instrucciones:
- Identifica automáticamente los datos clave presentes en el documento sin necesidad de una lista predefinida
- Devuelve un JSON estructurado con los datos extraídos y una breve justificación para cada uno, explicando por qué es relevante
- Si algún dato no está presente o no se puede inferir con certeza, exclúyelo del JSON.
- No inventes información ni realices suposiciones
- Devuelve únicamente el JSON, sin texto adicional

Ejemplo de salida:
{
"propietario": { "valor": "Nombre Apellido", "razon": "Figura en el apartado de titularidad" },
"referencia_catastral": { "valor": "123456789", "razon": "Se indica como referencia del inmueble" },
"ubicación": { "valor": "Calle X, Ciudad", "razon": "Direccion completa del inmueble" }
}
"""

system_prompt_agente_validador = """
Eres un experto en verificación de documentos legales y notas simples de viviendas.
Tu tarea es revisar la información extraída de la nota simple y validar que sea correcta y coherente con el documento original,
y si es necesario, modificar algun campo.

Explicacion de comentarios:
- Ok: Los datos estan bien.
- Incongruencia: Una incongruencia será cuando el tipo del valor no tenga sentido con el campo, por ejemplo un número de telefono con textos
o un nombre propio con numeros.
- Dato incorrecto: El dato que extrae el extractor no coincide con el dato del texto.

Instrucciones:
- Comparar: Verifica que los datos extraídos coincidan con la información presente en la nota simple.
- Precisión: Revisa que los valores extraídos sean correctos y no haya errores de interpretación.
- Formato: Confirma que la información esté bien estructurada en JSON.
- Corrección: Si encuentras un valor incongruente, ponlo a null y cambia el estado de correcto a true y pon como comentario
incongruencia. Toma como ejemplo el siguiente json que te entraria:
"propietario": { "valor": "134", "razon": "Figura en el apartado de titularidad" }
Toma como ejemplo lo que tienes que sacar:
"propietario": { "valor": null, "correcto": true, "comentario": "Incongruencia"}

Devuelve un JSON estructurado con cada campo validado, donde:
- Cada campo incluirá su valor extraído, una validación booleana (true si es correcto, false si es incorrecto), y un comentario.
- Si un campo es incorrecto o incongruente, proporciona un comentario breve explicando el error.
- Si un campo es correcto, ponle como comentario Ok.
- Si un campo no está en la nota simple original, no lo incluyas en el JSON.
- No generes texto adicional fuera del JSON.

Ejemplo de salida:
{
"propietario": { "valor": "Nombre Apellido", "correcto": true, "comentario": "Ok" },
"referencia_catastral": { "valor": "123456789", "correcto": false, "comentario": "Dato incorrecto" },
"ubicación": { "valor": "Calle X, Ciudad", "correcto": true, "comentario": "Ok" },
"codigo_postal": { "valor": null, "correcto": true, "comentario": "Incongruencia" }
}
"""

system_prompt_agente_supervisor = """
Eres un supervisor de agentes LLM encargados de extraer y validar información de notas simples de viviendas. Tu tarea es coordinar el trabajo entre el agente extractor y el agente validador, asegurando que el proceso sea recursivo en caso de errores y se obtenga una salida precisa y validada.

Instrucciones:

Evaluar la validación:
- Revisa el JSON generado por el agente validador para identificar campos con validado: false.

Retroalimentación al extractor:
- Si encuentras campos incorrectos o incompletos pero no incongruentes recibidos por el validador, solicita al
agente extractor que refine únicamente los datos problemáticos.

Revalidar:
- Una vez corregidos los campos, vuelve a ejecutar el agente validador para verificar las modificaciones.
- Repite el proceso hasta que todos los campos correcto tengan el valor true.
- Si recibes un mensaje del tipo { "valor": null, "correcto": true, "comentario": "Incongruencia" }, tómalo como correcto.

Finalizar:
- Solo aprueba la información cuando todos los datos sean correctos.
- Devuelve un JSON final con los datos clave ya validados, incluyendo un campo "status": "ok" si todos los datos son correctos.
- Si persisten errores tras la iteracion, devuelve "status": "error" y especifica los campos problemáticos.

Ejemplo de salida correcta:
{
"propietario": { "valor": "Nombre Apellido", "correcto": true, "comentario": "Ok" },
"referencia_catastral": { "valor": "123456789", "correcto": true, "comentario": "Ok" },
"ubicación": { "valor": "Calle X, Ciudad", "correcto": true, "comentario": "Ok" },
"codigo_postal": { "valor": null, "correcto": true, "comentario": "Incongruencia" },
"status": "ok"
}

Ejemplo de salida erronea:
{
"propietario": { "valor": "Nombre Apellido", "correcto": true, "comentario": "Ok" },
"referencia_catastral": { "valor": "12345", "correcto": false, "comentario": "Dato incorrecto" },
"status": "error"
}
"""

class Router(BaseModel):
    """Trabajador al que dirigir a continuación. Si no se necesitan trabajadores, dirige a FINISH."""

    next: Literal["extractor", "validador", "FINISH"]  # Lista explícita de estados
    motivo: str = Field(description="Motivo por el cual has decidido que trabajador viene a continiuación")


# Definimos al supervisor como un nodo
def supervisor_node(state: MessagesState) -> Command[Literal["extractor", "validador", "__end__"]]:
    """
    El supervisor invoca al LLM y decide a quién pasar el control.
    Si el LLM responde FINISH, salta a __end__.
    """
    messages = [
        {"role": "system", "content": system_prompt_agente_supervisor},
    ] + state["messages"]

    # Invocación con structured_output => Parse en un modelo Pydantic
    logging.info(f"\n===== ENTRADA SUPERVISOR =====\n{state['messages'][-1].content.encode().decode('utf-8')}\n=============================")
    log_messages.append(f"\n=======================================\nENTRADA SUPERVISOR\n\n{state['messages'][-1].content.encode().decode('utf-8')}\n=======================================")
    response = llm_supervisor.with_structured_output(Router).invoke(messages)
    logging.info(f"\n===== SALIDA SUPERVISOR =====\nSiguiente nodo: {response.next}\nRazón: {response.motivo}\n=============================")
    log_messages.append(f"\n=======================================\nSALIDA SUPERVISOR\n\nSiguiente nodo: {response.next}\nRazón: {response.motivo}\n=======================================")

    # Obtener siguiente acción
    goto = response.next  # Equivalente a response.dict()["next"]

    # Si el LLM ordena FINISH, forzamos el end
    if goto == "FINISH":
        return Command(goto="__end__")

    # De lo contrario, pasamos al nodo devuelto
    return Command(goto=goto)

agente_extractor = create_react_agent(
    model=llm_trabajador,
    tools=[],
    state_modifier=SystemMessage(content=system_prompt_agente_extractor),
)

agente_validador = create_react_agent(
    model=llm_trabajador,
    tools=[],
    state_modifier=SystemMessage(content=system_prompt_agente_validador),
)


# Nodo extractor
def extractor_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    logging.info(f"\n===== ENTRADA EXTRACTOR =====\n{state['messages'][-1].content.encode().decode('utf-8')}\n============================")
    log_messages.append(f"\n=======================================\nENTRADA EXTRACTOR\n\n{state['messages'][-1].content.encode().decode('utf-8')}\n=======================================")
    result = agente_extractor.invoke(state)
    response_content = result["messages"][-1].content
    logging.info(f"\n===== SALIDA EXTRACTOR =====\n{response_content}\n==============================")
    log_messages.append(f"\n=======================================\nSALIDA EXTRACTOR\n\n{response_content}\n=======================================")


    return Command(
        update={
            "messages": [
                HumanMessage(content=response_content, name="extractor")
            ]
        },
        goto="supervisor",
    )


# Nodo validador
def validador_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    logging.info(f"\n===== ENTRADA VALIDADOR =====\n{state['messages'][-1].content.encode().decode('utf-8')}\n============================")
    log_messages.append(f"\n=======================================\nENTRADA VALIDADOR\n\n{state['messages'][-1].content.encode().decode('utf-8')}\n=======================================")
    result = agente_validador.invoke(state)
    response_content = result["messages"][-1].content
    logging.info(f"\n===== SALIDA VALIDADOR =====\n{response_content}\n==============================")
    log_messages.append(f"\n=======================================\nSALIDA VALIDADOR\n\n{response_content}\n=======================================")


    return Command(
        update={
            "messages": [
                HumanMessage(content=response_content, name="validador")
            ]
        },
        goto="supervisor",
    )

logging.info("Iniciando construcción del grafo")
# Construimos el grafo
builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")

builder.add_node("supervisor", supervisor_node)
builder.add_node("extractor", extractor_node)
builder.add_node("validador", validador_node)

config = {"recursion_limit": 10}
graph = builder.compile()
logging.info("Construcción del grafo finalizada")


def extract_information(text):
    try:
        list_messages = []
        global log_messages  # noqa: PLW0603
        log_messages = ["Iniciando procesamiento de extracción.",]

        yield None, "\n".join(log_messages)

        for s in graph.stream(
            {"messages": [HumanMessage(content=text)]},
            subgraphs=True,
            config=config
        ):
            list_messages.append(s)
            time.sleep(1)  # Simula el tiempo de procesamiento

            # Actualizar solo la sección de logs en tiempo real
            yield None, "\n".join(log_messages)

        if len(list_messages) >= 2:  # Asegurar que haya al menos dos elementos  # noqa: PLR2004
                ultimo_mensaje = list_messages[-2][-1]
                if "validador" in ultimo_mensaje:
                    json_string = list_messages[-2][-1]["validador"]["messages"][0].content
                    json_string = extract_json(json_string)

                    # Mensaje final en logs
                    log_messages.append("Información extraída correctamente.")
                    upload_json_to_gpc(json_string)

                    # Finalmente, actualizar el JSON en la UI
                    yield json_string, "\n".join(log_messages)
                    return

        # Si el último mensaje no es del validador
        log_messages.append("No se ha extraído la información correctamente.")
        yield None, "\n".join(log_messages)

    except Exception as e:
        error_message = f"Error en la extracción: {e!s}"
        log_messages.append(error_message)
        logging.exception("Error al acceder a los datos del mensaje")
        yield None, "\n".join(log_messages)

with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    /* Main container styling */
    .gradio-container {
        max-width: 700px;
        margin: auto;
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
    }
    /* Styling for buttons */
    #transcribe_btn, #extract_btn {
        background-color: #A100FF;
        color: #FFFFFF;
        font-weight: bold;
    }
    h1 { color: white; }
    .title {
        text-align: center;
        width: 100%;
    }
    #logo {
        margin-bottom: -4vh;
    }
    /* Fixed height columns using Flexbox */
    .fixed-height-row {
        display: flex;
    }
    .center-row {
        display: flex;
        justify-content: center; /* This centers the row content horizontally */
    }
    .fixed-height-column {
        display: flex;
        flex-direction: column;
        height: 78vh;        /* Fixed height for each column */
        overflow: auto;       /* Scrollbar if content exceeds the fixed height */
        padding: 10px;
        box-sizing: border-box;
    }
    .grow-container {
        flex: 1;              /* Fill available vertical space */
        display: flex;
        flex-direction: column;
        min-height: 0;
        width: 100%;
    }
    """
) as app:

    # First row: centered logo .
    with gr.Row(elem_classes="center-row"):
        gr.Image(
            value="accenturelogo.png",
            show_label=False,
            container=False,
            height="6vh",
            width="40vw",
            elem_id="logo",
            show_download_button=False,
            show_fullscreen_button=False,
        )

    # Second row: centered title.
    with gr.Row(elem_classes="center-row"):
        gr.Markdown(
            "<div style='text-align: center;'><h1 style='color:#A100FF'>Procesamiento de notas simples</h1></div>",
            elem_id="title"
        )

    # Third row: three fixed-height columns.
    with gr.Row(elem_classes="fixed-height-row"):
        # Column 1: File upload, extracted text and "Transcribir" button.
        with gr.Column(elem_id="col1", elem_classes="fixed-height-column"):
            file_input = gr.File(label="Sube tu archivo PDF", file_types=[".pdf"], file_count="single")
            with gr.Column(elem_classes="grow-container"):
                text_output = gr.Textbox(label="Texto extraído", interactive=False,autofocus=False, lines=23)
            transcribe_button = gr.Button("Transcribir", elem_id="transcribe_btn")

        # Column 2: JSON output and "Extraer información" button.
        with gr.Column(elem_id="col2", elem_classes="fixed-height-column"):
            with gr.Column(elem_classes="grow-container"):
                json_output = gr.JSON(label="JSON Obtenido", min_height="69vh")
            extract_button = gr.Button("Extraer información", elem_id="extract_btn")

        # Column 3: New textbox for "Comunicación entre agentes".
        with gr.Column(elem_id="col3", elem_classes="fixed-height-column"):  # noqa: SIM117
            with gr.Column(elem_classes="grow-container"):
                communication_box = gr.Textbox(label="Comunicación entre agentes", interactive=False, lines=30)

    # Define button callbacks.
    transcribe_button.click(fn=extract_text_from_pdf, inputs=file_input, outputs=text_output)
    extract_button.click(fn=extract_information, inputs=text_output, outputs=[json_output, communication_box])



app.launch(share=True, server_name="localhost", server_port=8081)


