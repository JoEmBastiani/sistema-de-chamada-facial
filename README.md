# Sistema de Chamada por Reconhecimento Facial

Sistema web completo para automatizar a chamada de alunos usando reconhecimento facial com OpenCV, Face Recognition e Deep Learning.

## Funcionalidades

- Cadastro de alunos com foto
- Detecção facial usando Haar Cascade
- Reconhecimento facial usando Face Recognition
- Chamada automatizada via webcam
- Interface web intuitiva
- API REST para integração

## Tecnologias Utilizadas

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Visão Computacional**: OpenCV, Face Recognition

## Pré-requisitos

- Python 3.8 ou superior
- Webcam funcional
- Sistema operacional: Windows/Linux/Mac

## Instalação

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd facial-attendance-system
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Instale o CMake (necessário para face_recognition)
#### Windows:
```bash
pip install cmake
```
#### Linux (Ubuntu/Debian):
```bash
sudo apt install cmake
```
#### Mac:
```bash
brew install cmake
```

## Como usar

### 1. Inicie o backend
```bash
python backend.py
```
O servidor backend iniciará em: http://localhost:8000

### 2. Inicie o frontend
Em outro terminal:
```bash
streamlit run app.py
```
O frontend abrirá automaticamente em: http://localhost:8501

### 3. Fluxo de Uso

#### Cadastrar Alunos:

- Acesse "Registrar Aluno" no menu
- Digite o nome do aluno
- Faça upload da foto ou tire uma foto com a câmera
- Clique em "Registrar Aluno"

#### Realizar Chamada:

- Acesse "Realizar Chamada"
- Permita o acesso à câmera
- O sistema detectará e reconhecerá faces automaticamente

#### Ver Presenças:

- Acesse "Ver Presenças"
- Visualize a lista de presença
- Exporte para CSV se necessário

## Funcionamento Técnico

### Detecção Facial (Haar Cascade)

- Utiliza haarcascade_frontalface_default.xml do OpenCV
- Detecta faces em tempo real na webcam
- Desenha bounding boxes nas faces detectadas

### Reconhecimento Facial (Face Recognition)

- Extrai embeddings faciais de 128 dimensões
- Compara com banco de dados de alunos cadastrados
- Utiliza tolerância de 0.6 para matching
- Calcula distância euclidiana entre embeddings

### API Endpoints

- `POST /register-student` - Registra novo aluno
- `POST /recognize-faces` - Reconhece faces em imagem
- `GET /attendance - Retorna` lista de presença
- `POST /reset-attendance` - Reseta chamada
- `GET /students` - Lista alunos cadastrados