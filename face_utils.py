import face_recognition
import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, images_path: str = "student_images"):
        self.images_path = images_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_list = {}
        self.encodings_file = "encodings.pkl"
        
        # Criar diretório se não existir
        os.makedirs(images_path, exist_ok=True)
        
        # Carregar encodings existentes
        self.load_encodings()
    
    def load_encodings(self):
        """Carrega encodings salvos ou cria novos"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
        else:
            self.train_model()
    
    def train_model(self):
        """Treina o modelo com as imagens dos alunos"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.images_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Carregar imagem
                image_path = os.path.join(self.images_path, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Obter encodings da face
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    # Usar o primeiro encoding encontrado
                    self.known_face_encodings.append(face_encodings[0])
                    # Extrair nome do aluno do nome do arquivo
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
        
        # Salvar encodings
        with open(self.encodings_file, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        
        print(f"Modelo treinado com {len(self.known_face_names)} alunos")
    
    def detect_faces_haarcascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta faces usando Haar Cascade"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        return faces
    
    def recognize_faces(self, image: np.ndarray) -> List[dict]:
        """Reconhece faces em uma imagem"""
        # Converter BGR para RGB (face_recognition usa RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar faces
        face_locations = self.detect_faces_haarcascade(image)
        
        # Obter encodings das faces detectadas
        face_encodings = face_recognition.face_encodings(rgb_image, 
                                                         [(y, x+w, y+h, x) for (x, y, w, h) in face_locations])
        
        results = []
        
        for (x, y, w, h), face_encoding in zip(face_locations, face_encodings):
            name = "Desconhecido"
            confidence = 0
            
            # Comparar com faces conhecidas
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        
                        # Registrar presença
                        if name not in self.attendance_list:
                            self.attendance_list[name] = {
                                'time': datetime.now().strftime("%H:%M:%S"),
                                'date': datetime.now().strftime("%Y-%m-%d")
                            }
            
            results.append({
                'name': name,
                'confidence': float(confidence),
                'location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        return results
    
    def register_student(self, name: str, image_path: str) -> bool:
        """Registra um novo aluno"""
        try:
            # Copiar imagem para o diretório de alunos
            filename = f"{name}.jpg"
            destination = os.path.join(self.images_path, filename)
            
            # Carregar e salvar imagem
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Detectar face na imagem
            faces = self.detect_faces_haarcascade(image)
            if len(faces) == 0:
                return False
            
            # Salvar apenas a região da face
            x, y, w, h = faces[0]
            face_image = image[y:y+h, x:x+w]
            cv2.imwrite(destination, face_image)
            
            # Retreinar o modelo
            self.train_model()
            
            return True
        except Exception as e:
            print(f"Erro ao registrar aluno: {e}")
            return False
    
    def get_attendance_list(self) -> dict:
        """Retorna a lista de presença"""
        return self.attendance_list
    
    def reset_attendance(self):
        """Reseta a lista de presença"""
        self.attendance_list = {}