from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
from face_utils import FaceRecognitionSystem
import shutil

app = FastAPI(title="Sistema de Reconhecimento Facial")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema de reconhecimento
face_system = FaceRecognitionSystem()

@app.get("/")
async def root():
    return {"message": "API de Reconhecimento Facial - Sistema de Chamada"}

@app.post("/register-student")
async def register_student(name: str, file: UploadFile = File(...)):
    """Registra um novo aluno com sua foto"""
    try:
        # Criar diretório temporário
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Salvar arquivo temporariamente
        temp_path = os.path.join(temp_dir, f"{name}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Registrar aluno
        success = face_system.register_student(name, temp_path)
        
        # Limpar arquivo temporário
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if success:
            return JSONResponse(
                content={"message": f"Aluno {name} registrado com sucesso!"},
                status_code=200
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="Não foi possível detectar uma face na imagem fornecida"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize-faces")
async def recognize_faces(file: UploadFile = File(...)):
    """Reconhece faces em uma imagem enviada"""
    try:
        # Ler imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Imagem inválida")
        
        # Reconhecer faces
        results = face_system.recognize_faces(image)
        
        return JSONResponse(content={
            "faces_detected": results,
            "total_faces": len(results)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/attendance")
async def get_attendance():
    """Retorna a lista de presença atual"""
    attendance = face_system.get_attendance_list()
    return JSONResponse(content={
        "attendance": attendance,
        "total_present": len(attendance)
    })

@app.post("/reset-attendance")
async def reset_attendance():
    """Reseta a lista de presença"""
    face_system.reset_attendance()
    return JSONResponse(content={"message": "Lista de presença resetada com sucesso!"})

@app.get("/students")
async def get_students():
    """Retorna a lista de alunos cadastrados"""
    students = face_system.known_face_names
    return JSONResponse(content={
        "students": students,
        "total_students": len(students)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)