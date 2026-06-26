import streamlit as st
import requests
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Sistema de Chamada Facial",
    page_icon="👤",
    layout="wide"
)

# URL da API
API_URL = "http://localhost:8000"

class VideoProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.last_capture_time = 0
        self.capture_interval = 3  # segundos
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Detectar faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Desenhar retângulos
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Capturar e enviar frame periodicamente
        current_time = time.time()
        if current_time - self.last_capture_time >= self.capture_interval:
            self.last_capture_time = current_time
            self.send_frame_to_api(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def send_frame_to_api(self, frame):
        """Envia frame para API de reconhecimento"""
        try:
            # Converter frame para bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                response = requests.post(f"{API_URL}/recognize-faces", files=files)
        except Exception as e:
            pass

def main():
    st.title("Sistema de Chamada por Reconhecimento Facial")
    st.markdown("---")
    
    # Menu lateral
    with st.sidebar:
        st.header("Menu")
        menu_option = st.radio(
            "Selecione uma opção:",
            ["Registrar Aluno", "Realizar Chamada", "Ver Presenças", "Alunos Cadastrados"]
        )
        
        if st.button("Reiniciar Chamada"):
            try:
                response = requests.post(f"{API_URL}/reset-attendance")
                if response.status_code == 200:
                    st.success("Chamada reiniciada")
            except:
                st.error("Erro ao reiniciar chamada")
    
    # Conteúdo principal
    if menu_option == "Registrar Aluno":
        st.header("Registrar Novo Aluno")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            student_name = st.text_input("Nome do Aluno:", key="student_name")
            
            # Opção de upload ou câmera
            upload_method = st.radio(
                "Método de captura:",
                ["Upload de Foto", "Usar Câmera"]
            )
            
            if upload_method == "Upload de Foto":
                uploaded_file = st.file_uploader(
                    "Escolha uma foto do aluno",
                    type=['jpg', 'jpeg', 'png']
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Preview", use_column_width=True)
                    
                    if st.button("Registrar Aluno", type="primary"):
                        if student_name:
                            with st.spinner("Registrando aluno..."):
                                try:
                                    files = {'file': uploaded_file.getvalue()}
                                    response = requests.post(
                                        f"{API_URL}/register-student",
                                        params={"name": student_name},
                                        files=files
                                    )
                                    
                                    if response.status_code == 200:
                                        st.success(f"Aluno {student_name} registrado com sucesso!")
                                    else:
                                        st.error("Erro: Face não detectada na imagem")
                                except Exception as e:
                                    st.error(f"Erro de conexão: {str(e)}")
                        else:
                            st.warning("Por favor, insira o nome do aluno")
            
            else:
                st.info("Use a câmera para capturar a foto do aluno")
                camera_image = st.camera_input("Tire uma foto")
                
                if camera_image is not None:
                    if st.button("Registrar Aluno", type="primary"):
                        if student_name:
                            with st.spinner("Registrando aluno..."):
                                try:
                                    files = {'file': camera_image.getvalue()}
                                    response = requests.post(
                                        f"{API_URL}/register-student",
                                        params={"name": student_name},
                                        files=files
                                    )
                                    
                                    if response.status_code == 200:
                                        st.success(f"Aluno {student_name} registrado com sucesso!")
                                    else:
                                        st.error("Erro: Face não detectada na imagem")
                                except Exception as e:
                                    st.error(f"Erro de conexão: {str(e)}")
                        else:
                            st.warning("Por favor, insira o nome do aluno")
    
    elif menu_option == "Realizar Chamada":
        st.header("Chamada Automática")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("Câmera em funcionamento - Detectando faces automaticamente...")
            
            # Componente de webcam
            webrtc_ctx = webrtc_streamer(
                key="face-recognition",
                video_processor_factory=VideoProcessor,
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            st.caption("As faces detectadas são enviadas automaticamente para reconhecimento")
        
        with col2:
            st.subheader("Status da Chamada")

            # Atualização automática
            placeholder = st.empty()
            while True:
                try:
                    response = requests.get(f"{API_URL}/attendance")
                    if response.status_code == 200:
                        data = response.json()
                        with placeholder.container():
                            st.metric("Alunos Presentes", data['total_present'])
                except:
                    pass
                time.sleep(2)
    
    elif menu_option == "Ver Presenças":
        st.header("Lista de Presença")
        
        # Data atual
        current_date = datetime.now().strftime("%d/%m/%Y")
        st.subheader(f"Data: {current_date}")
        
        try:
            response = requests.get(f"{API_URL}/attendance")
            if response.status_code == 200:
                data = response.json()
                
                if data['total_present'] > 0:
                    # Criar tabela de presença
                    attendance_table = []
                    for name, info in data['attendance'].items():
                        attendance_table.append({
                            "Aluno": name,
                            "Horário": info['time'],
                            "Status": "Presente"
                        })
                    
                    # Mostrar tabela
                    st.table(attendance_table)
                    
                    # Exportar dados
                    if st.button("Exportar CSV"):
                        import pandas as pd
                        df = pd.DataFrame(attendance_table)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"presenca_{current_date.replace('/','_')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Nenhuma presença registrada ainda")
            else:
                st.error("Erro ao carregar lista de presença")
        except Exception as e:
            st.error(f"Erro de conexão: {str(e)}")
    
    elif menu_option == "Alunos Cadastrados":
        st.header("Alunos Cadastrados")
        
        try:
            response = requests.get(f"{API_URL}/students")
            if response.status_code == 200:
                data = response.json()
                
                if data['total_students'] > 0:
                    st.success(f"Total de {data['total_students']} alunos cadastrados")
                    
                    # Mostrar lista em cards
                    cols = st.columns(3)
                    for idx, student in enumerate(data['students']):
                        with cols[idx % 3]:
                            st.info(f"**{student}**")
                else:
                    st.warning("Nenhum aluno cadastrado")
            else:
                st.error("Erro ao carregar alunos")
        except Exception as e:
            st.error(f"Erro de conexão: {str(e)}")

if __name__ == "__main__":
    main()