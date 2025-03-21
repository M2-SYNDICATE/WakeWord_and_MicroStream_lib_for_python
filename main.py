import time
from wakeword import AudioStream, Detection  # Импортируем классы из вашей библиотеки

# Создаем экземпляр AudioStream
stream = AudioStream(
    buffer_size=1024,      # Размер буфера (можно настроить)
    desired_channels=1,    # Моно (1 канал)
    desired_sample_rate=16000  # Частота дискретизации 16000 Гц
)

# Запускаем аудиопоток
print("Запуск аудиопотока...")
stream.start()

# Загружаем модель (укажите путь к вашему .pmdl файлу)
model_path = "./xander.rpw"  # Замените на реальный путь
print(f"Загрузка модели из {model_path}...")
stream.load_model(model_path)

# Основной цикл обработки аудио
print("Начало обработки аудио. Говорите активационное слово!")
try:
    while True:
        # Получаем аудио-чанк из потока
        chunk = stream.get_audio_chunk()
        
        # Если чанк не пустой, передаем его на детекцию
        if chunk:
            detections = stream.detect(chunk)
            
            # Обрабатываем результаты детекции
            for detection in detections:
                if detection is not None:
                    print(f"Обнаружено: {detection.name} с уверенностью {detection.score}")

                # else:
                #     print("Активационное слово не обнаружено")
        
        # Небольшая задержка для предотвращения перегрузки CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nОстановка программы пользователем.")
finally:
    # Здесь можно добавить код для остановки потока, если потребуется
    pass
