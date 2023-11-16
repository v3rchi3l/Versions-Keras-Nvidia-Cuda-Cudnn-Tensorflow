import tensorflow as tf
import subprocess

def get_gpu_memory():
    try:
        # Ejecutar el comando 'nvidia-smi' para obtener información de la GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

        if result.returncode == 0:
            total_memory = int(result.stdout.strip())  # Memoria total de la GPU en MB
            return total_memory

    except FileNotFoundError:
        print("nvidia-smi no encontrado. Asegúrate de tener instalados los controladores de NVIDIA.")

    return None

# Obtener la memoria total de la GPU
total_gpu_memory = get_gpu_memory()
if total_gpu_memory is not None:
    batch_size_percentage = 0.7
    batch_size_70 = int(total_gpu_memory * batch_size_percentage)

    print(f"Total de memoria VRAM de la GPU: {total_gpu_memory:.2f} MB")
    print(f"Tamaño del lote para utilizar el 70% de VRAM: {batch_size_70} MB")
else:
    print("No se pudo obtener la memoria de la GPU.")

def press_any_key_to_continue():
    input("Presione cualquier tecla para continuar... El siguiente paso es TEST de tensorflow con GPU")

# Llamar a la función para mostrar el mensaje y esperar la entrada del usuario
press_any_key_to_continue()


# Verificar la disponibilidad de GPU
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))

# Crear un modelo sencillo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(10,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generar datos de ejemplo
import numpy as np
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size_70)