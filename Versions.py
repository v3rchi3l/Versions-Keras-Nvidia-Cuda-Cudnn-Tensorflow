import torch
import tensorflow as tf
import subprocess
import os

print("=" * 40)
print("=" * 40)
# Ejecutar el comando nvidia-smi
resultado = subprocess.run(['nvidia-smi'], shell=True, capture_output=True, text=True)

# Imprimir la salida del comando nvidia-smi
print(resultado.stdout)

print("=" * 40)
print("=" * 40)
# Ejecutar el comando nvcc --version para obtener la versión de CUDA
resultado = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)

# Capturar la salida del comando e imprimir la versión de CUDA
salida = resultado.stdout
lineas = salida.split('\n')
version_cuda = None

for linea in lineas:
    if 'release' in linea.lower():
        version_cuda = linea.strip()
        break

if version_cuda:
    print("Versión de CUDA instalada:", version_cuda)
else:
    print("CUDA no está instalado o no se pudo encontrar la versión.")

directorio_cudnn = r'C:\Program Files\NVIDIA\CUDNN'
print("=" * 40)
print("=" * 40)
# Verificar si el directorio de cuDNN existe
if os.path.exists(directorio_cudnn) and os.path.isdir(directorio_cudnn):
    # Obtener la lista de archivos y carpetas en el directorio cuDNN
    contenido_cudnn = os.listdir(directorio_cudnn)

    # Filtrar solo los nombres de carpetas (versiones)
    versiones_cudnn = [elemento for elemento in contenido_cudnn if os.path.isdir(os.path.join(directorio_cudnn, elemento))]

    # Imprimir las versiones de cuDNN encontradas
    if versiones_cudnn:
        print("Versiones de cuDNN encontradas: " + ' - '.join(versiones_cudnn))
    else:
        print("No se encontraron versiones de cuDNN en el directorio especificado.")
else:
    print("El directorio de cuDNN no existe o no es un directorio válido.")

# Obtener la variable de entorno PATH
path_env = os.getenv('PATH')

if path_env:
    # Dividir las diferentes rutas utilizando el separador adecuado según el sistema operativo
    if os.name == 'nt':  # En sistemas Windows, las rutas están separadas por ";"
        paths = path_env.split(';')
    else:  # En sistemas Unix/Linux, las rutas están separadas por ":"
        paths = path_env.split(':')

    # Filtrar las rutas que contienen "cudnn"
    paths_containing_cudnn = [ruta for ruta in paths if "cudnn" in ruta.lower()]

    # Mostrar las rutas que contienen "cudnn"
    if paths_containing_cudnn:
        print("Rutas que contienen 'cudnn' en la variable PATH:")
        for ruta in paths_containing_cudnn:
            print(ruta)
    else:
        print("No se encontraron rutas que contengan 'cudnn' en la variable PATH.")
else:
    print("La variable de entorno PATH no está definida.")

print("=" * 40)
print("=" * 40)
print("detalles de Tensorflow")
# Imprimir la versión de TensorFlow
print("Versión de TensorFlow:", tf.__version__)

# Imprimir la versión de Keras (si estás usando TensorFlow 2.x)
print("Versión de Keras:", tf.keras.__version__)

# Obtener la lista de dispositivos físicos GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Si se encontró al menos una GPU
    for i, gpu in enumerate(gpus):
        print(f"Se encontró una GPU en la ubicación {i}: {gpu}")
else:
    # Si no se encontraron GPUs
    print("No se encontró ninguna GPU.")
print("=" * 40)
print("=" * 40)
print("Detalles de Pytorch")
# Imprimir la versión de PyTorch
print("version de Pytorch" + torch.__version__)

print("Verifica la disponibilidad de CUDA (GPU) en PyTorch")
print(torch.cuda.is_available())

print("Muestra el número de dispositivos CUDA disponibles")
print(torch.cuda.device_count())

print("Muestra información detallada sobre la GPU")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # Si tienes varias GPUs, cambia el índice (0) según sea necesario
    print(torch.cuda.get_device_properties(0))  # Muestra propiedades detalladas del dispositivo
